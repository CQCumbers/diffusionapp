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
    config.computeUnits = cpuOnly ? MLComputeUnitsCPUOnly : MLComputeUnitsAll;
    return [MLModel modelWithContentsOfURL:url configuration:config error:err];
}

static MLMultiArray *array_init(float *data, NSArray *shape, NSArray *strides) {
    NSError *err = NULL;
    MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:data
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    return array;
}

static void print_array(const float *array, const int *strides, int indent) {
    /* pretty print an MLMultiArray */
    int inner = strides[1] == 1, dot_i = inner ? 0 : indent + 2;
    char sep = inner ? ' ' : '\n';
    printf("%*s[%c", indent, "", sep);
    for (int i = 0, len = strides[0] / strides[1]; i < len; ++i) {
        if (i >= 3 && i < len - 3) printf("%*s...%c", dot_i, "", sep), i = len - 3;
        const float *elem = array + i * strides[1];
        if (inner) printf("%f ", *elem);
        else print_array(elem, strides + 1, indent + 2);
    }
    printf("%*s]\n", inner ? 0 : indent, "");
}

static void print_stats(const float *array, int size) {
    float mean = 0, mean2 = 0;
    for (int i = 0; i < size; ++i) {
        mean = mean + (array[i] - mean) / (i + 1);
        mean2 = mean2 + (array[i] * array[i] - mean2) / (i + 1);
    }
    printf("mean: %f stddev %f\n", mean, mean2 - mean * mean);
}

/* === PNDM scheduler === */

static void sched_init(float *alphas, float start, float end, int steps) {
    float beta_start = sqrtf(start);
    float beta_step = (sqrtf(end) - beta_start) / (1000 - 1);
    float alpha = 1.0 - start;
    alphas[steps] = alpha;
    for (int step = steps, n = 0; step-- > 0;) {
        for (int i = 0; i < 1000 / steps; ++i) {
            if (i == 1) alphas[step] = alpha;
            float beta = beta_start + beta_step * ++n;
            alpha *= (float)(1.0 - beta * beta);
        }
    }
}

static void sched_step(const float *alphas, int step,
        const float *e_t, float *latents, int size) {
    int idx = step ? step - 1 : step; // 0-1 use same alphas
    float a_t = alphas[idx + 0], b_t = 1.0 - a_t;
    float a_p = alphas[idx + 1], b_p = 1.0 - a_p;

    float l_coeff = sqrtf(a_p / a_t);
    float n_denom = a_t * sqrtf(b_p) + sqrtf(a_t * b_t * a_p);
    float n_coeff = (a_p - a_t) / n_denom;

    const float weights[5][4] = {
        { 1.0 }, { 0.5, 0.5 }, { 1.5, -0.5 },
        { 23/12.0, -16/12.0, 5/12.0 },
        { 55/24.0, -59/24.0, 37/24.0, -9/24.0 }
    };

    int w_idx = step < 4 ? step : 4;
    const float *n[4], *w = weights[w_idx];
    for (int t = 0; t < 4; ++t)
        n[t] = e_t + (step + 4 - t) % 4 * size;

    printf("alphas %.8f %.8f coeffs %.8f %.8f\n", a_t, a_p, l_coeff, n_coeff);
    for (int i = 0; i < size; ++i) {
        float avg = w[0] * n[0][i] + w[1] * n[1][i];
        avg += w[2] * n[2][i] + w[3] * n[3][i];
        latents[i] = l_coeff * latents[i] - n_coeff * avg;
    }
}

static void transpose_embeds(const float *embeds, float *hidden) {
    for (int i = 0; i < 77; ++i) {
        for (int j = 0; j < 768; ++j) {
            hidden[j * 77 + i] = embeds[i * 768 + j];
        }
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
    float *hidden;
    float *latents;
    float *step;
    float *out;
    float *e_t;
    float *image;
    float *alphas;
    unsigned rng[4];
} buffers_t;

static model_t *encoder_init(MLModel *model, float *ids, float *embeds) {
    model_t *enc = malloc(sizeof(model_t));
    NSError *err = NULL;
    enc->model = model ? model : load_model(@"text_encoder", 0, &err);
    if (!enc->model) return free(enc), NULL;

    MLMultiArray *enc_ids = array_init(ids, @[ @1, @77 ], @[ @77, @1 ]);
    enc->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input_ids":enc_ids } error:&err];
    if (!enc->inputs) return free(enc), NULL;

    MLMultiArray *enc_embeds = array_init(embeds, @[ @77, @768 ], @[ @768, @1 ]);
    enc->options = [[MLPredictionOptions alloc] init];
    [enc->options setOutputBackings:@{ @"last_hidden_state":enc_embeds }];
    return enc;
}

static model_t *diffuser_init(MLModel *model,
        float *hidden, float *step, float *latents, float *preds) {
    model_t *dif = malloc(sizeof(model_t));
    NSError *err = NULL;
    dif->model = model ? model : load_model(@"unet", 0, &err);
    if (!dif->model) return free(dif), NULL;

    NSArray *shape = @[ @2, @4, @64, @64 ], *strides = @[ @16384, @4096, @64, @1 ];
    MLMultiArray *dif_hidden = array_init(hidden, @[ @2, @768, @1, @77 ], @[ @59136, @77, @77, @1 ]);
    MLMultiArray *dif_step = array_init(step, @[ @2 ], @[ @1 ]);
    MLMultiArray *dif_latents = array_init(latents, shape, strides);
    dif->inputs = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"sample":dif_latents, @"timestep":dif_step, @"encoder_hidden_states":dif_hidden} error:&err];
    if (!dif->inputs) return free(dif), NULL;

    MLMultiArray *dif_preds = array_init(preds, shape, strides);
    dif->options = [[MLPredictionOptions alloc] init];
    [dif->options setOutputBackings:@{ @"noise_pred":dif_preds }];
    return dif;
}

static model_t *decoder_init(MLModel *model, float *z, float *image) {
    model_t *dec = malloc(sizeof(model_t));
    NSError *err = NULL;
    dec->model = model ? model : load_model(@"vae_decoder", 0, &err);
    if (!dec->model) return free(dec), NULL;

    NSArray *shape = @[ @1, @4, @64, @64 ], *strides = @[ @16384, @4096, @64, @1 ];
    MLMultiArray *dec_z = array_init(z, shape, strides);
    dec->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"z":dec_z } error:&err];
    if (!dec->inputs) return free(dec), NULL;

    MLMultiArray *dec_image = array_init(image, @[ @3, @512, @512 ], @[ @262144, @512, @1 ]);
    dec->options = [[MLPredictionOptions alloc] init];
    [dec->options setOutputBackings:@{ @"image":dec_image }];
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
    print_array(data.embeds, embeds_s, 0);
    transpose_embeds(data.embeds, data.hidden + 768 * 77);

    /* Setup scheduler alphas */
    t2i_request_t *req = t2i_request(engine, req_id);
    if (req->steps > 127) req->steps = 127;
    data.step[0] = data.step[1] = 1001 - 1000 % req->steps;
    sched_init(data.alphas, 0.00085, 0.0120, req->steps);

    /* Generate latent noise vector */
    const int n_noise = 4 * 64 * 64;
    float *latents = data.latents, *out = data.out;
    xoshiro_seed(data.rng, req->seed);
    for (int i = 0; i < n_noise; i += 2)
        gaussians_next(data.rng, latents + i);

    const int max_filename = 128;
    char filename[max_filename];
    float guidance = req->guide / 10.0f;
    for (int step = 0; step <= req->steps; ++step) {
        /* run UNet to predict noise residual */
        if (step != 2) data.step[0] = data.step[1] -= 1000 / req->steps;
        NSLog(@"Running step %d: %f\n", step, data.step[0]);
        if (handle(ctx, req_id, T2I_STEPS + step)) return 1;
        memcpy(latents + n_noise, latents, n_noise * sizeof(float));

        [dif->model predictionFromFeatures:dif->inputs options:dif->options error:&err];
        if (err != NULL) {
            NSLog(@"UNet error: %@\n", [err localizedDescription]);
            return handle(ctx, req_id, T2I_UNET_FAILED);
        }

        /* Perform guidance */
        float *e_t = data.e_t + step % 4 * n_noise;
        for (int i = 0; i < n_noise; ++i)
            e_t[i] = out[i] + guidance * (out[n_noise + i] - out[i]);

        /* Compute previous noisy sample */
        sched_step(data.alphas, step, data.e_t, latents, n_noise);
        for (int i = 0; i < n_noise; ++i)
            out[i] = latents[i] / 0.18215;

        /* Decode latents into image */
        [dec->model predictionFromFeatures:dec->inputs options:dec->options error:&err];
        if (err != NULL) {
            NSLog(@"Decoder error: %@\n", [err localizedDescription]);
            return handle(ctx, req_id, T2I_DECODER_FAILED);
        }

        for (int i = 0; i < 512 * 512; ++i) {
            for (int j = 0; j < 3; ++j) {
                float pixel = fmin(fmax(data.image[j * 512 * 512 + i], -1), 1);
                req->image[i * 3 + j] = (unsigned char)(pixel * 127 + 128);
            }
        }
        snprintf(filename, max_filename, "step_%d.png", step);
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
    NSString *path = [[NSBundle mainBundle] pathForResource:@"bpe_context" ofType:@"bin"];
    bpe_context_t tokenizer = bpe_init([path UTF8String]);
    if (!tokenizer) return handle(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    bpe_encode(tokenizer, "", ids, 77);

    /* Load encoder model and buffers */
    NSArray *shape, *strides;
    data.embeds = calloc(77 * 768, sizeof(float));
    model_t *enc = encoder_init(NULL, ids, data.embeds);
    if (!enc) return handle(ctx, -1, T2I_ENCODER_NOLOAD), NULL;
    if (handle(ctx, -1, T2I_ENCODER_LOADED)) return NULL;

    NSError *err = NULL;
    [enc->model predictionFromFeatures:enc->inputs options:enc->options error:&err];
    if (err != NULL) {
        NSLog(@"Encoder error: %@\n", [err localizedDescription]);
        return handle(ctx, -1, T2I_ENCODER_FAILED), NULL;
    }

    /* Load diffuser model and buffers */
    const int n_noise = 4 * 64 * 64;
    data.hidden = calloc(2 * 768 * 77, sizeof(float));
    data.latents = calloc(2 * n_noise, sizeof(float));
    data.step = calloc(2, sizeof(float));
    data.out = calloc(2 * n_noise, sizeof(float));

    model_t *dif = diffuser_init(NULL, data.hidden, data.step, data.latents, data.out);
    if (!dif) return handle(ctx, -1, T2I_UNET_NOLOAD), NULL;
    if (handle(ctx, -1, T2I_UNET_LOADED)) return NULL;
    transpose_embeds(data.embeds, data.hidden);

    /* Load decoder model and buffers */
    data.image = calloc(3 * 512 * 512, sizeof(float));
    data.alphas = calloc(128, sizeof(float));
    data.e_t = calloc(4 * n_noise, sizeof(float));

    model_t *dec = decoder_init(NULL, data.out, data.image);
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
