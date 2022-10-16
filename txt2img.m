#import "txt2img.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <pthread.h>
#import <stdatomic.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#import "stb_image_write.h"
#import "bpe.h"

/* === CoreML helpers === */

static MLModel *load_model(NSString *name, int cpuOnly, NSError **err) {
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
    config.computeUnits = cpuOnly ? MLComputeUnitsCPUOnly : MLComputeUnitsAll;
    return [MLModel modelWithContentsOfURL:cache_url
        configuration:config error:err];
}

static MLMultiArray *array_init(float *data, NSArray *shape, NSArray *strides) {
    NSError *err = NULL;
    MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:data
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    return array;
}

static void print_array(float *array, int *strides, int indent) {
    /* pretty print an MLMultiArray */
    int inner = strides[1] == 1, dot_i = inner ? 0 : indent + 2;
    char sep = inner ? ' ' : '\n';
    printf("%*s[%c", indent, "", sep);
    for (int i = 0, len = strides[0] / strides[1]; i < len; ++i) {
        if (i >= 3 && i < len - 3) printf("%*s...%c", dot_i, "", sep), i = len - 3;
        float *elem = array + i * strides[1];
        if (inner) printf("%f ", *elem);
        else print_array(elem, strides + 1, indent + 2);
    }
    printf("%*s]\n", inner ? 0 : indent, "");
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
   int head, tail;
   pthread_cond_t empty;
   pthread_mutex_t mutex;
   pthread_t worker;
   t2i_handler_t handler;
   void *context;
};

static void *worker_main(void *args);

t2i_t t2i_init(t2i_handler_t handler, void *ctx) {
    /* Allocate engine data */
    t2i_t engine = malloc(sizeof(struct T2IEngine));
    engine->handler = handler, engine->context = ctx;
    engine->head = 0, engine->tail = 0;
    pthread_cond_init(&engine->empty, NULL);
    pthread_mutex_init(&engine->mutex, NULL);

    /* Start worker thread */
    int res = pthread_create(&engine->worker, NULL, worker_main, engine);
    if (res) return NSLog(@"pthread error: %d", res), NULL;
    return engine;
}

void t2i_free(t2i_t engine) {
    /* Wait for worker to finish */
    pthread_join(engine->worker, NULL);
    free(engine);
}

t2i_request_t *t2i_request(t2i_t engine, int req_id) {
    /* Assume valid and access synchronized */
    assert(req_id != -1 && "Can't fetch invalid req_id");
    return &engine->queue[req_id % N_QUEUE];
}

int t2i_acquire(t2i_t engine) {
    /* Fail if too many pending requests */
    pthread_mutex_lock(&engine->mutex);
    int head = engine->head, tail = engine->tail;
    pthread_mutex_unlock(&engine->mutex);
    return head < tail + N_QUEUE ? head : -1;
}

void t2i_submit(t2i_t engine, int req_id) {
    /* Increment pending requests (req_id from acquire) */
    pthread_mutex_lock(&engine->mutex);
    int head = engine->head++;
    pthread_cond_signal(&engine->empty);
    pthread_mutex_unlock(&engine->mutex);
    assert(req_id == head && "Can't submit requests out of order");
}

static int queue_front(t2i_t engine) {
    /* Wait for pending request */
    pthread_mutex_lock(&engine->mutex);
    int tail = engine->tail;
    while (tail >= engine->head)
        pthread_cond_wait(&engine->empty, &engine->mutex);
    pthread_mutex_unlock(&engine->mutex);
    return tail;
}

static void queue_pop(t2i_t engine, int req_id) {
    /* Increment released requests (req_id from front) */
    pthread_mutex_lock(&engine->mutex);
    int tail = engine->tail++;
    pthread_mutex_unlock(&engine->mutex);
    assert(req_id == tail && "Can't release requests out of order");
}

/* === Model implementation === */

typedef struct {
    MLModel *model;
    MLDictionaryFeatureProvider *inputs;
    MLPredictionOptions *options;
} model_t;

static model_t *encoder_init(MLModel *model, float *ids, float *embeds) {
    model_t *enc = malloc(sizeof(model_t));
    NSError *err = NULL;
    enc->model = model ? model : load_model(@"text_encoder", 0, &err);
    if (!enc->model) return free(enc), NULL;

    MLMultiArray *enc_ids = array_init(ids, @[ @1, @77 ], @[ @77, @1 ]);
    enc->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input_ids_1":enc_ids } error:&err];
    if (!enc->inputs) return free(enc), NULL;

    MLMultiArray *enc_embeds = array_init(embeds, @[ @77, @768 ], @[ @768, @1 ]);
    enc->options = [[MLPredictionOptions alloc] init];
    [enc->options setOutputBackings:@{ @"last_hidden_state":enc_embeds }];
    return enc;
}

static model_t *unet_init(MLModel *model,
        float *embeds, float *step, float *latents, float *preds) {
    model_t *unet = malloc(sizeof(model_t));
    NSError *err = NULL;
    unet->model = model ? model : load_model(@"unet", 0, &err);
    if (!unet->model) return free(unet), NULL;

    NSArray *shape = @[ @2, @4, @64, @64 ], *strides = @[ @16384, @4096, @64, @1 ];
    MLMultiArray *u_embeds = array_init(embeds, @[ @2, @77, @768 ], @[ @59136, @768, @1 ]);
    MLMultiArray *u_step = array_init(step, @[ @1 ], @[ @1 ]);
    MLMultiArray *u_latents = array_init(latents, shape, strides);
    unet->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"sample_1":u_latents, @"timestep":u_step, @"input_35":u_embeds } error:&err];
    if (!unet->inputs) return free(unet), NULL;

    MLMultiArray *u_preds = array_init(preds, shape, strides);
    unet->options = [[MLPredictionOptions alloc] init];
    [unet->options setOutputBackings:@{ @"var_5609":u_preds }];
    return unet;
}

static model_t *quant_init(MLModel *model, float *preds, float *z) {
    model_t *quant = malloc(sizeof(model_t));
    NSError *err = NULL;
    quant->model = model ? model : load_model(@"post_quant_conv", 0, &err);
    if (!quant->model) return free(quant), NULL;

    NSArray *shape = @[ @1, @4, @64, @64 ], *strides = @[ @4096, @4096, @64, @1 ];
    MLMultiArray *quant_preds = array_init(preds, shape, strides);
    quant->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input":quant_preds } error:&err];
    if (!quant->inputs) return free(quant), NULL;

    MLMultiArray *quant_z = array_init(z, shape, strides);
    quant->options = [[MLPredictionOptions alloc] init];
    [quant->options setOutputBackings:@{ @"var_22":quant_preds }];
    return quant;
}

static model_t *decoder_init(MLModel *model, float *z, float *image) {
    model_t *dec = malloc(sizeof(model_t));
    NSError *err = NULL;
    dec->model = model ? model : load_model(@"vae_decoder", 0, &err);
    if (!dec->model) return free(dec), NULL;

    NSArray *shape = @[ @1, @4, @64, @64 ], *strides = @[ @4096, @4096, @64, @1 ];
    MLMultiArray *dec_z = array_init(z, shape, strides);
    dec->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"z":dec_z } error:&err];
    if (!dec->inputs) return free(dec), NULL;

    MLMultiArray *dec_image = array_init(image, @[ @3, @512, @512 ], @[ @262144, @512, @1 ]);
    dec->options = [[MLPredictionOptions alloc] init];
    [dec->options setOutputBackings:@{ @"var_730":dec_image }];
    return dec;
}

static void *worker_main(void *args) {
    t2i_t engine = args;
    void *ctx = engine->context;

    /* Initialize tokenizer */
    const int n_ids = 77;
    float *ids = calloc(n_ids, sizeof(float));
    bpe_context_t tokenizer = bpe_init("vocab.json", "merges.txt");
    bpe_encode(tokenizer, "", ids, n_ids);

    /* Load encoder model and buffers */
    NSArray *shape, *strides;
    float *embeds = calloc(2 * 77 * 768, sizeof(float));

    model_t *enc = encoder_init(NULL, ids, embeds + 77 * 768);
    if (!enc) return engine->handler(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    model_t *uncond = encoder_init(enc->model, ids, embeds);
    if (!uncond) return engine->handler(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    if (engine->handler(ctx, -1, T2I_ENCODER_LOADED)) return NULL;

    NSError *err = NULL;
    [uncond->model predictionFromFeatures:uncond->inputs options:uncond->options error:&err];
    if (err) return engine->handler(ctx, -1, T2I_ENCODER_FAILED), NULL;

    /* Load unet model and buffers */
    const int n_noise = 4 * 64 * 64, max_steps = 128;
    float (*e_t)[n_noise] = calloc(2 * n_noise, sizeof(float));
    float (*latents)[n_noise] = calloc(2 * n_noise, sizeof(float));
    float step = 0.0f;

    model_t *unet = unet_init(NULL, embeds, &step, (float*)latents, (float*)e_t);
    if (!unet) return engine->handler(ctx, -1, T2I_UNET_NOLOAD), NULL;
    if (engine->handler(ctx, -1, T2I_UNET_LOADED)) return NULL;

    /* Load decoder model and buffers */
    const int max_filename = 64;
    float *decoded = calloc(3 * 512 * 512, sizeof(float));
    float alphas[max_steps + 1];
    unsigned rng[4];
    char filename[max_filename];

    model_t *quant = quant_init(NULL, e_t[0], e_t[1]);
    if (!quant) return engine->handler(ctx, -1, T2I_DECODER_NOLOAD), NULL;
    model_t *dec = decoder_init(NULL, e_t[1], decoded);
    if (!dec) return engine->handler(ctx, -1, T2I_DECODER_NOLOAD), NULL;
    if (engine->handler(ctx, -1, T2I_DECODER_LOADED)) return NULL;

    for (;;) {
        /* Get request to process */
        int req_id = queue_front(engine);
        t2i_request_t *req = t2i_request(engine, req_id);
        bpe_encode(tokenizer, req->prompt, ids, n_ids);

        /* Get encoder output embeddings */
        [enc->model predictionFromFeatures:enc->inputs options:enc->options error:&err];
        if (err) engine->handler(ctx, req_id, T2I_ENCODER_FAILED);

        printf("\nEncoder output:\n");
        int embeds_s[] = { 77 * 768, 768, 1 };
        print_array(embeds + 77 * 768, embeds_s, 0);

        /* Generate latent noise vector */
        if (req->steps > max_steps) req->steps = max_steps;
        sched_init(alphas, 0.00085, 0.0120, req->steps);
        xoshiro_seed(rng, req->seed);
        for (int i = 0; i < n_noise; i += 2)
            gaussians_next(rng, &latents[0][i]);

        for (int i = 0; i < req->steps; ++i, step -= 1000 / req->steps) {
            /* run UNet to predict noise residual */
            NSLog(@"Running step %d: %f\n", i, step);
            if (engine->handler(ctx, req_id, T2I_STEPS + i)) break;
            memcpy(latents[1], latents[0], n_noise * sizeof(float));

            printf("\nUNet Input:\n");
            print_array(embeds, embeds_s, 0);
            printf("\nLatent values:\n");
            int latents_s[] = { 4 * 64 * 64, 64 * 64, 64, 1 };
            print_array(latents[0], latents_s, 0);

            [unet->model predictionFromFeatures:unet->inputs options:unet->options error:&err];
            if (err) engine->handler(ctx, req_id, T2I_UNET_FAILED);

            /* Perform guidance */
            /* TODO: implement variable guide/strength */
            for (int i = 0; i < n_noise; ++i)
                e_t[0][i] += 7.5 * (e_t[1][i] - e_t[0][i]);

            printf("\n\ne_t values:\n");
            print_array((float*)e_t, latents_s, 0);

            /* Compute previous noisy sample */
            sched_step(alphas, i, e_t[0], latents[0], n_noise);
            for (int i = 0; i < n_noise; ++i)
                e_t[0][i] = 1.0 / 0.18215 * latents[0][i];

            /* Decode latents into image */
            [quant->model predictionFromFeatures:quant->inputs options:quant->options error:&err];
            if (err) engine->handler(ctx, req_id, T2I_UNET_FAILED);
            [dec->model predictionFromFeatures:dec->inputs options:dec->options error:&err];
            if (err) engine->handler(ctx, req_id, T2I_UNET_FAILED);

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
        engine->handler(ctx, req_id, T2I_FINISHED);
        queue_pop(engine, req_id);
    }

    bpe_free(tokenizer);
}

/* === Command line tests === */

/*static int logger(void *ctx, int req_id, int status) {
    printf("Status %d for %d\n", status, req_id);
    return 0;
}

int main (int argc, char *argv[]) {
    if (argc < 2) return puts("Usage: txt2img <text>");
    t2i_t engine = t2i_init(logger, NULL);

    int req_id = t2i_acquire(engine);
    t2i_request_t *req = t2i_request(engine, req_id);
    strncpy(req->prompt, argv[1], N_TOKENS);
    req->steps = 21, req->seed = 123;

    t2i_submit(engine, req_id);
    t2i_free(engine);
    return 0;
}*/
