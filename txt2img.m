#import "txt2img.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <pthread.h>
#import <stdatomic.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#import "stb_image_write.h"
#import "bpe.h"

/* === CoreML helpers === */

static MLModel *load_model(NSString *name, int device, NSError **err) {
    NSFileManager *manager = [NSFileManager defaultManager];
    NSString *cache = [NSString stringWithFormat:@"%@.mlmodelc", name];
    NSLog(@"Loading %@.mlmodelc", name);

    if (![manager fileExistsAtPath:cache]) {
        NSLog(@"Compiling from .mlmodel\n");
        NSString *src = [NSString stringWithFormat:@"%@.mlmodel", name];
        NSURL *src_url = [NSURL fileURLWithPath:src];
        NSURL *tmp = [MLModel compileModelAtURL:src_url error:err];
        [manager copyItemAtPath:tmp.path toPath:cache error:err];
    }

    NSURL *cache_url = [NSURL fileURLWithPath:cache];
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = device ? MLComputeUnitsAll : MLComputeUnitsCPUOnly;
    return [MLModel modelWithContentsOfURL:cache_url
        configuration:config error:err];
}

static void print_array(float *array, int *strides, int indent) {
    /* pretty print an MLMultiArray */
    for (int i = 0; i < indent; ++i) printf(" ");
    printf("[ ");
    for (int i = 0, len = strides[0] / strides[1]; i < len; ++i) {
        if (i >= 3 && i < len - 3) continue;
        float *elem = array + i * strides[1];
        if (strides[1] == 1) printf("%f ", *elem);
        else print_array(elem, strides + 1, indent + 2);
    }
    for (int i = 0; i < indent; ++i) printf(" ");
    printf("]\n");
}

/* === DDIM? scheduler === */

static void sched_init(float *alphas, float start, float end, int steps) {
    float beta_start = sqrtf(start);
    float beta_step = (sqrtf(end) - beta_start) / 1000;
    float alpha = 1.0 - start, beta = beta_start;
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < 1000 / steps; ++i) {
            if (i == 1) alphas[step + 1] = alpha;
            beta += beta_step, alpha *= 1.0 - beta * beta;
        }
    }
    alphas[0] = 1.0;
}

static void sched_step(const float *alphas, int i,
        const float *noise, float *latents, int size) {
    float alpha_t = alphas[i + 1], alpha_p = alphas[i];
    float sqrt_at = sqrtf(alpha_t), sqrt_ap = sqrtf(alpha_p);
    float sqrt_1at = sqrtf(1.0 - alpha_t);
    float sqrt_1ap = sqrtf(1.0 - alpha_p);

    for (int i = 0; i < size; ++i) {
        float pred_x0 = latents[i] - sqrt_1at * noise[i];
        latents[i] = (pred_x0 / sqrt_at) * sqrt_ap + sqrt_1ap * noise[i];
    }
}

/* === Noise generation === */

static void xoshiro_seed(unsigned *s, unsigned seed) {
    /* Seed with Splitmix32 */
    for (unsigned i = 0; i < 4; ++i) {
        unsigned z = (seed += 0x9e3779b9);
        z = (z ^ (z >> 16)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        s[i] = z ^ (z >> 16);
    }
}

static float xoshiro_next(unsigned *s) {
    /* Sample with Xoshiro128+ */
    unsigned result = s[0] + s[3];
    unsigned t = s[1] << 9;

    s[2] ^= s[0], s[3] ^= s[1];
    s[1] ^= s[2], s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >> 21);
    return (result >> 8) * 0x1.0p-24;
}

static void gaussians_next(unsigned *s, float *out) {
    /* Apply Marsaglia Polar Method */
    float u, v, r = 0.0;
    while (r == 0.0 || r >= 1.0) {
        u = xoshiro_next(s) * 2 - 1;
        v = xoshiro_next(s) * 2 - 1;
        r = u * u + v * v;
    }
    r = sqrtf(-2 * logf(r) / r);
    out[0] = u * r, out[1] = v * r;
}

/* === Engine interface === */

#define N_QUEUE 3

struct T2IEngine {
   t2i_request_t queue[N_QUEUE];
   atomic_int head, tail;
   pthread_t worker;
   t2i_handler_t handler;
   void *context;
};

static void *worker_main(void *args);

t2i_t t2i_init(t2i_handler_t handler, void *ctx) {
    /* Allocate engine data and worker thread */
    t2i_t engine = calloc(1, sizeof(struct T2IEngine));
    int res = pthread_create(&engine->worker, NULL, worker_main, engine);
    if (res) return NSLog(@"pthread error: %d", res), NULL;

    /* Store callback function */
    engine->handler = handler;
    engine->context = ctx;
    return engine;
}

void t2i_free(t2i_t engine) {
    /* Wait for worker to finish */
    pthread_join(engine->worker, NULL);
    free(engine);
}

t2i_request_t *t2i_request(t2i_t engine, int req_id) {
    /* Assume valid and access synchronized */
    return &engine->queue[req_id % N_QUEUE];
}

int t2i_acquire(t2i_t engine) {
    /* Avoid overwriting pending requests */
    int head = atomic_load_explicit(&engine->head, memory_order_relaxed);
    int tail = atomic_load_explicit(&engine->tail, memory_order_acquire);
    return head < tail + N_QUEUE ? head : -1;
}

void t2i_submit(t2i_t engine, int req_id) {
    /* Increment pending requests */
    int head = atomic_load_explicit(&engine->head, memory_order_relaxed);
    int tail = atomic_load_explicit(&engine->tail, memory_order_acquire);
    assert(req_id == head && "Can't submit requests out of order");
    atomic_store_explicit(&engine->head, head + 1, memory_order_release);
}

static int queue_front(t2i_t engine) {
    /* Retrieve pending request */
    int tail = atomic_load_explicit(&engine->tail, memory_order_relaxed);
    int head = atomic_load_explicit(&engine->head, memory_order_acquire);
    return tail < head ? tail : -1;
}

static void queue_pop(t2i_t engine, int req_id) {
    /* Increment released requests */
    int tail = atomic_load_explicit(&engine->tail, memory_order_relaxed);
    int head = atomic_load_explicit(&engine->head, memory_order_acquire);
    assert(req_id == tail && "Can't release requests out of order");
    atomic_store_explicit(&engine->tail, tail + 1, memory_order_release);
}

/* === Model implementation === */

static void *worker_main(void *args) {
    t2i_t engine = args;
    void *ctx = engine->context;

    /* Initialize tokenizer */
    const int n_ids = 77;
    float *ids = calloc(n_ids, sizeof(float));
    bpe_context_t tokenizer = bpe_init("vocab.json", "merges.txt");
    bpe_encode(tokenizer, "", ids, n_ids);

    /* Load encoder model and inputs */
    NSError *err = nil;
    MLModel *encoder = load_model(@"text_encoder", 1, &err);
    if (!encoder) return engine->handler(ctx, -1, T2I_ENCODER_NOLOAD);

    NSArray *shape = @[ @1, @(n_ids) ], *strides = @[ @(n_ids), @1 ];
    MLMultiArray *encoder_ids = [[MLMultiArray alloc] initWithDataPointer:ids
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLDictionaryFeatureProvider* encoder_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input_ids_1":encoder_ids } error:&err];
    if (!encoder_in) return engine->handler(ctx, -1, T2I_ENCODER_NOLOAD);

    id<MLFeatureProvider> encoder_out = [encoder predictionFromFeatures:encoder_in error:&err];
    if (!encoder_out) return engine->handler(ctx, -1, T2I_ENCODER_FAILED);
    MLMultiArray *uncond = [[encoder_out featureValueForName:@"last_hidden_state"] multiArrayValue];

    NSLog(@"encoder successfully loaded");
    if (engine->handler(ctx, -1, T2I_ENCODER_LOADED)) return 1;

    /* Load unet model and inputs */
    const int n_noise = 4 * 64 * 64;
    float (*latents)[n_noise] = calloc(2 * n_noise, sizeof(float));
    float *rescaled = calloc(n_noise, sizeof(float));
    float *step = calloc(1, sizeof(float));
    MLModel *unet = load_model(@"unet", 1, &err);
    if (!unet) return engine->handler(ctx, -1, T2I_UNET_NOLOAD);

    shape = @[ @2, @4, @64, @64 ], strides = @[ @16384, @4096, @64, @1 ];
    MLMultiArray *unet_latents = [[MLMultiArray alloc] initWithDataPointer:latents
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLMultiArray *unet_step = [[MLMultiArray alloc] initWithDataPointer:step
        shape:@[ @1 ] dataType:MLMultiArrayDataTypeFloat32 strides:@[ @1 ]
        deallocator:^(void *bytes) { (void)bytes; } error:&err];

    NSLog(@"unet successfully loaded");
    if (engine->handler(ctx, -1, T2I_UNET_LOADED)) return 1;

    /* Load VAE and prepare inputs */
    MLModel *post_quant = load_model(@"post_quant_conv", 1, &err);
    if (!post_quant) return engine->handler(ctx, -1, T2I_DECODER_NOLOAD);
    MLModel *decoder = load_model(@"vae_decoder", 1, &err);
    if (!decoder) return engine->handler(ctx, -1, T2I_DECODER_NOLOAD);

    shape = @[ @1, @4, @64, @64 ], strides = @[ @4096, @4096, @64, @1 ];
    MLMultiArray *quant_latents = [[MLMultiArray alloc] initWithDataPointer:rescaled
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLDictionaryFeatureProvider* post_quant_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input":quant_latents } error:&err];

    NSLog(@"decoder successfully loaded");
    if (engine->handler(ctx, -1, T2I_DECODER_LOADED)) return 1;

    const int max_filename = 64, max_steps = 128;
    float alphas[max_steps + 1];
    char filename[max_filename];
    unsigned rng[4];

    for (;;) {
        /* Get request to process */
        int req_id = queue_front(engine);
        t2i_request_t *req = t2i_request(engine, req_id);
        bpe_encode(tokenizer, req->prompt, ids, N_TOKENS);

        /* Get encoder output embeddings */
        encoder_out = [encoder predictionFromFeatures:encoder_in error:&err];
        if (!encoder_out) engine->handler(ctx, req, T2I_ENCODER_FAILED);
        MLMultiArray *embeds = [[encoder_out featureValueForName:@"last_hidden_state"] multiArrayValue];

        printf("\nEncoder output:\n");
        int embeds_s[] = { 77 * 768, 768, 1 };
        print_array(embeds.dataPointer, embeds_s, 0);

        MLMultiArray *unet_embeds = [MLMultiArray
            multiArrayByConcatenatingMultiArrays:@[ uncond, embeds ]
            alongAxis:0 dataType:MLMultiArrayDataTypeFloat32];
        MLDictionaryFeatureProvider* unet_in = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary: @{ @"sample_1":unet_latents, @"timestep":unet_step, @"input_35":unet_embeds }
            error:&err];
        if (!unet_in) engine->handler(ctx, req, T2I_UNET_NOLOAD);

        /* Generate latent noise vector */
        if (req->steps > max_steps) req->steps = max_steps;
        sched_init(alphas, 0.00085, 0.0120, req->steps);
        xoshiro_seed(rng, req->seed);
        for (int i = 0; i < n_noise; i += 2)
            gaussians_next(rng, &latents[0][i]);

        for (int i = 0; i < req->steps; ++i, step -= 1000 / req->steps) {
            /* run UNet to predict noise residual */
            NSLog(@"Running step %d: %f\n", i, step);
            if (engine->handler(ctx, req, T2I_STEPS + step)) break;
            memcpy(latents[1], latents[0], n_noise * sizeof(float));

            printf("\nUNet Input:\n");
            print_array(unet_embeds.dataPointer, embeds_s, 0);
            printf("\nLatent values:\n");
            int latents_s[] = { 4 * 64 * 64, 64 * 64, 64, 1 };
            print_array(latents[0], latents_s, 0);

            id<MLFeatureProvider> unet_out = [unet predictionFromFeatures:unet_in error:&err];
            if (!unet_out) engine->handler(ctx, req, T2I_UNET_FAILED);
            MLMultiArray *pred = [[unet_out featureValueForName:@"var_5609"] multiArrayValue];

            /* Perform guidance */
            float (*e_t)[n_noise] = pred.dataPointer;
            for (int i = 0; i < n_noise; ++i)
                e_t[0][i] += 7.5 * (e_t[1][i] - e_t[0][i]);

            printf("\n\ne_t values:\n");
            print_array(e_t, latents_s, 0);

            /* Compute previous noisy sample */
            sched_step(alphas, i, e_t[0], latents[0], n_noise);
            for (int i = 0; i < n_noise; ++i)
                rescaled[i] = 1.0 / 0.18215 * latents[0][i];

            /* Decode latents into image */
            id<MLFeatureProvider> post_quant_out = [post_quant predictionFromFeatures:post_quant_in error:&err];
            if (!post_quant_out) engine->handler(ctx, req, T2I_DECODER_FAILED);
            MLMultiArray *z = [[post_quant_out featureValueForName:@"var_22"] multiArrayValue];
            MLDictionaryFeatureProvider *decoder_in = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{ @"z":z } error:&err];

            id<MLFeatureProvider> decoder_out = [decoder predictionFromFeatures:decoder_in error:&err];
            if (!decoder_out) engine->handler(ctx, req, T2I_DECODER_FAILED);
            MLMultiArray *decoder_z = [[decoder_out featureValueForName:@"var_730"] multiArrayValue];
            float *decoded = decoder_z.dataPointer;

            for (int i = 0; i < 512 * 512; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float pixel = decoded[j * 512 * 512 + i];
                    req->image[i * 3 + j] = (unsigned char)(pixel * 127 + 128);
                }
            }
            snprintf(filename, max_filename, "step_%d.png", i);
            stbi_write_png(filename, 512, 512, 3, req->image, 512 * 3);
        }

        /* Notify request finished */
        engine->handler(ctx, req, T2I_FINISHED);
        queue_pop(engine, req_id);
    }

    bpe_free(tokenizer);
}

/* === Command line tests === */

static int logger(void *ctx, t2i_request_t *req, int status) {
    /* log error states */
    printf("Logger status %d\n", status);
    return 0;
}

int main (int argc, char *argv[]) {
    if (argc < 2) return puts("Usage: txt2img <text>");
    t2i_t engine = t2i_init(logger, NULL);

    int req_id = t2i_acquire(engine);
    t2i_request_t *req = t2i_request(engine, req_id);
    strncpy(req->prompt, argv[1], N_TOKENS);
    req->steps = 21, req->seed = 123;
    /* implement guide/seed */

    t2i_submit(engine, req_id);
    /* wait for finish signal */
    t2i_free(engine);
    return 0;
}
