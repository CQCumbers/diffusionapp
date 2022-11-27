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
    NSLog(@"Loading %@ from mlmodelc\n", name);
    NSURL *url = [[NSBundle mainBundle] URLForResource:name withExtension:@"mlmodelc"];
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = cpuOnly ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndGPU;
    return [MLModel modelWithContentsOfURL:url configuration:config error:err];
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

typedef struct {
    float *embeds;
    float *latents;
    float *e_t;
    float *image;
    float *alphas;
    unsigned rng[4];
    float step;
} buffers_t;

static model_t *encoder_init(MLModel *model, float *ids, float *embeds) {
    model_t *enc = malloc(sizeof(model_t));
    NSError *err = NULL;
    enc->model = model ? model : load_model(@"txt_encoder", 0, &err);
    if (!enc->model) return free(enc), NULL;

    MLMultiArray *enc_ids = array_init(ids, @[ @1, @77 ], @[ @77, @1 ]);
    enc->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"in_ids":enc_ids } error:&err];
    if (!enc->inputs) return free(enc), NULL;

    MLMultiArray *enc_embeds = array_init(embeds, @[ @77, @768 ], @[ @768, @1 ]);
    enc->options = [[MLPredictionOptions alloc] init];
    [enc->options setOutputBackings:@{ @"out_embeds":enc_embeds }];
    return enc;
}

static model_t *diffuser_init(MLModel *model,
        float *embeds, float *step, float *latents, float *preds) {
    model_t *dif = malloc(sizeof(model_t));
    NSError *err = NULL;
    dif->model = model ? model : load_model(@"diffuser", 0, &err);
    if (!dif->model) return free(dif), NULL;

    NSArray *shape = @[ @2, @4, @64, @64 ], *strides = @[ @16384, @4096, @64, @1 ];
    MLMultiArray *dif_embeds = array_init(embeds, @[ @2, @77, @768 ], @[ @59136, @768, @1 ]);
    MLMultiArray *dif_step = array_init(step, @[ @1 ], @[ @1 ]);
    MLMultiArray *dif_latents = array_init(latents, shape, strides);
    dif->inputs = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"in_latents":dif_latents, @"in_timestep":dif_step, @"in_embeds":dif_embeds} error:&err];
    if (!dif->inputs) return free(dif), NULL;

    MLMultiArray *dif_preds = array_init(preds, shape, strides);
    dif->options = [[MLPredictionOptions alloc] init];
    [dif->options setOutputBackings:@{ @"out_preds":dif_preds }];
    return dif;
}

static model_t *decoder_init(MLModel *model, float *z, float *image) {
    model_t *dec = malloc(sizeof(model_t));
    NSError *err = NULL;
    dec->model = model ? model : load_model(@"decoder", 0, &err);
    if (!dec->model) return free(dec), NULL;

    NSArray *shape = @[ @1, @4, @64, @64 ], *strides = @[ @4096, @4096, @64, @1 ];
    MLMultiArray *dec_z = array_init(z, shape, strides);
    dec->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"in_z":dec_z } error:&err];
    if (!dec->inputs) return free(dec), NULL;

    MLMultiArray *dec_image = array_init(image, @[ @3, @512, @512 ], @[ @262144, @512, @1 ]);
    dec->options = [[MLPredictionOptions alloc] init];
    [dec->options setOutputBackings:@{ @"out_image":dec_image }];
    return dec;
}

static int process_request(t2i_t engine, int req_id, model_t *enc,
        model_t *dif, model_t *dec, buffers_t data) {
    t2i_handler_t handle = engine->handler;
    void *ctx = engine->context;

    /* Get encoder output embeddings */
    NSError *err = NULL;
    [enc->model predictionFromFeatures:enc->inputs options:enc->options error:&err];
    if (err != NULL) {
        NSLog(@"Encoder error: %@\n", [err localizedDescription]);
        return handle(ctx, req_id, T2I_ENCODER_FAILED);
    }

    NSLog(@"\nEncoder output:\n");
    int embeds_s[] = { 77 * 768, 768, 1 };
    print_array(data.embeds + 77 * 768, embeds_s, 0);

    /* Setup scheudler alphas */
    t2i_request_t *req = t2i_request(engine, req_id);
    if (req->steps > 127) req->steps = 127;
    sched_init(data.alphas, 0.00085, 0.0120, req->steps);

    /* Generate latent noise vector */
    const int n_noise = 4 * 64 * 64;
    float *latents = data.latents, *e_t = data.e_t;
    xoshiro_seed(data.rng, req->seed);
    for (int i = 0; i < n_noise; i += 2)
        gaussians_next(data.rng, &latents[i]);

    const int max_filename = 128;
    char filename[max_filename];
    float guidance = req->guide / 10.0f;
    for (int i = 0; i < req->steps; ++i, data.step -= 1000 / req->steps) {
        /* run UNet to predict noise residual */
        NSLog(@"Running step %d: %f\n", i, data.step);
        if (handle(ctx, req_id, T2I_STEPS + i)) return 1;
        memcpy(latents + n_noise, latents, n_noise * sizeof(float));

        NSLog(@"\nUNet Input:\n");
        print_array(data.embeds, embeds_s, 0);
        NSLog(@"\nLatent values:\n");
        int latents_s[] = { 4 * 64 * 64, 64 * 64, 64, 1 };
        print_array(latents, latents_s, 0);

        [dif->model predictionFromFeatures:dif->inputs options:dif->options error:&err];
        if (err != NULL) {
            NSLog(@"UNet error: %@\n", [err localizedDescription]);
            return handle(ctx, req_id, T2I_UNET_FAILED);
        }

        /* Perform guidance */
        for (int i = 0; i < n_noise; ++i)
            e_t[i] += guidance * (e_t[n_noise + i] - e_t[i]);

        NSLog(@"\ne_t values:\n");
        print_array((float*)e_t, latents_s, 0);

        /* Compute previous noisy sample */
        sched_step(data.alphas, i, e_t, latents, n_noise);
        for (int i = 0; i < n_noise; ++i)
            e_t[i] = 1.0 / 0.18215 * latents[i];

        /* Decode latents into image */
        [dec->model predictionFromFeatures:dec->inputs options:dec->options error:&err];
        if (err != NULL) {
            NSLog(@"Decoder error: %@\n", [err localizedDescription]);
            return handle(ctx, req_id, T2I_DECODER_FAILED);
        }

        for (int i = 0; i < 512 * 512; ++i) {
            for (int j = 0; j < 3; ++j) {
                float pixel = data.image[j * 512 * 512 + i];
                req->image[i * 3 + j] = (unsigned char)(pixel * 127 + 128);
            }
        }
        snprintf(filename, max_filename, "step_%d.png", i);
        stbi_write_png(filename, 512, 512, 3, req->image, 512 * 3);
    }

    NSLog(@"Successfully finished processing");
    return 0;
}

static void *worker_main(void *args) {
    t2i_t engine = (t2i_t)args;
    t2i_handler_t handle = engine->handler;
    void *ctx = engine->context;
    buffers_t data;

    /* Initialize tokenizer */
    float *ids = calloc(77, sizeof(float));
    bpe_context_t tokenizer = bpe_init("vocab.json", "merges.txt");
    bpe_encode(tokenizer, "", ids, 77);

    /* Load encoder model and buffers */
    NSArray *shape, *strides;
    data.embeds = calloc(2 * 77 * 768, sizeof(float));

    model_t *enc = encoder_init(NULL, ids, data.embeds + 77 * 768);
    if (!enc) return handle(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    model_t *uncond = encoder_init(enc->model, ids, data.embeds);
    if (!uncond) return handle(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    if (handle(ctx, -1, T2I_ENCODER_LOADED)) return NULL;

    NSError *err = NULL;
    [uncond->model predictionFromFeatures:uncond->inputs options:uncond->options error:&err];
    if (err != NULL) {
        NSLog(@"Encoder error: %@\n", [err localizedDescription]);
        return handle(ctx, -1, T2I_ENCODER_FAILED), NULL;
    }

    /* Load diffuser model and buffers */
    const int n_noise = 4 * 64 * 64;
    data.latents = calloc(2 * n_noise, sizeof(float));
    data.e_t = calloc(2 * n_noise, sizeof(float));
    data.step = 0.0f;

    model_t *dif = diffuser_init(NULL, data.embeds, &data.step, data.latents, data.e_t);
    if (!dif) return handle(ctx, -1, T2I_UNET_NOLOAD), NULL;
    if (handle(ctx, -1, T2I_UNET_LOADED)) return NULL;

    /* Load decoder model and buffers */
    data.image = calloc(3 * 512 * 512, sizeof(float));
    data.alphas = calloc(128, sizeof(float));

    model_t *dec = decoder_init(NULL, data.e_t + n_noise, data.image);
    if (!dec) return handle(ctx, -1, T2I_DECODER_NOLOAD), NULL;
    if (handle(ctx, -1, T2I_DECODER_LOADED)) return NULL;

    for (;;) {
        /* Get request to process */
        int req_id = queue_front(engine), err = 0;
        t2i_request_t *req = t2i_request(engine, req_id);
        bpe_encode(tokenizer, req->prompt, ids, 77);
        process_request(engine, req_id, enc, dif, dec, data);

        /* Notify request finished */
        handle(ctx, req_id, T2I_FINISHED);
        queue_pop(engine, req_id);
    }

    bpe_free(tokenizer);
}

/* === Command line tests === */

/*static int logger(void *ctx, int req_id, int status) {
    NSLog(@"Status %d for %d\n", status, req_id);
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
