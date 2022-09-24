#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#import "stb_image_write.h"
#import "bpe.h"

static MLModel *load_model(NSString *name, NSError **err) {
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
    return [MLModel modelWithContentsOfURL:cache_url error:err];
}

static float *sched_init(float start, float end, int steps) {
    float beta_start = sqrtf(start);
    float beta_step = (sqrtf(end) - beta_start) / 1000;

    float *alphas = malloc((steps + 1) * sizeof(float));
    float alpha = 1.0 - start, beta = beta_start;
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < 1000 / steps; ++i) {
            if (i == 1) alphas[step + 1] = alpha;
            beta += beta_step, alpha *= 1.0 - beta * beta;
        }
    }

    alphas[0] = 1.0;
    return alphas;
}

static void sched_step(const float *alphas, int i,
        const float *noise, float *latents, int size) {
    float alpha_t = alphas[i + 1], alpha_p = alphas[i];
    float ratio = sqrtf(alpha_p) / sqrtf(alpha_t);
    float sqrt_1at = sqrtf(1.0 - alpha_t);
    float sqrt_1ap = sqrtf(1.0 - alpha_p);

    for (int i = 0; i < size; ++i) {
        float pred_x0 = latents[i] - sqrt_1at * noise[i];
        latents[i] = ratio * pred_x0 + sqrt_1ap * noise[i];
    }
}

/* splitmix32/xoshiro128+ */
static unsigned *xoshiro_init(unsigned seed) {
    unsigned *s = calloc(4, sizeof(unsigned));
    for (unsigned i = 0; i < 4; ++i) {
        unsigned z = (seed += 0x9e3779b9);
        z = (z ^ (z >> 16)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        s[i] = z ^ (z >> 16);
    }
    return s;
}

static float xoshiro_next(unsigned *s) {
    unsigned result = s[0] + s[3];
    unsigned t = s[1] << 9;

    s[2] ^= s[0], s[3] ^= s[1];
    s[1] ^= s[2], s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >> 21);
    return (result >> 8) * 0x1.0p-24;
}

/* marsaglia polar method */
static void gaussians_next(unsigned *s, float *out) {
    float u, v, r = 0.0;
    while (r == 0.0 || r >= 1.0) {
        u = xoshiro_next(s) * 2 - 1;
        v = xoshiro_next(s) * 2 - 1;
        r = u * u + v * v;
    }
    r = sqrtf(-2 * logf(r) / r);
    out[0] = u * r, out[1] = v * r;
}

/* pretty printer for MLMultiArrays? */

int main (int argc, char *argv[]) {
    if (argc < 2) return puts("Usage: txt2img <text>");

    /* Encode input text as token ids */
    const int capacity = 77;
    float *ids = calloc(capacity, sizeof(float));
    bpe_context_t tokenizer = bpe_init("vocab.json", "merges.txt");
    bpe_encode(tokenizer, argv[1], ids, capacity);

    /* Load encoder model */
    NSError *err = nil;
    MLModel *encoder = load_model(@"text_encoder", &err);
    if (!encoder) NSLog(@"Can't load encoder: %@\n", err);

    /* Prepare encoder inputs */
    NSArray *shape = @[ @1, @(capacity) ], *strides = @[ @(capacity), @1 ];
    MLMultiArray *encoder_ids = [[MLMultiArray alloc] initWithDataPointer:ids
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    if (!encoder_ids) NSLog(@"Can't make encoder_ids: %@\n", err);
    MLDictionaryFeatureProvider* encoder_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input_ids_1":encoder_ids } error:&err];
    if (!encoder_in) NSLog(@"Can't make encoder_in: %@\n", err);

    /* Get encoder output embeddings */
    id<MLFeatureProvider> encoder_out = [encoder predictionFromFeatures:encoder_in error:&err];
    if (!encoder_out) NSLog(@"Can't run encoder: %@\n", err);
    MLMultiArray *embeds = [[encoder_out featureValueForName:@"last_hidden_state"] multiArrayValue];

    /* Get unconditional embeddings (assumes guidance_scale > 1.0) */
    bpe_encode(tokenizer, "", ids, capacity);
    encoder_out = [encoder predictionFromFeatures:encoder_in error:&err];
    if (!encoder_out) NSLog(@"Can't run encoder: %@\n", err);
    MLMultiArray *uncond = [[encoder_out featureValueForName:@"last_hidden_state"] multiArrayValue];

    /* Generate latent noise vector */
    const int n_steps = 12, n_noise = 4 * 64 * 64;
    float *alphas = sched_init(0.00085, 0.0120, n_steps);

    float (*latents)[n_noise] = calloc(2 * n_noise, sizeof(float));
    unsigned *rng = xoshiro_init(123);
    for (int i = 0; i < n_noise; i += 2)
        gaussians_next(rng, &latents[0][i]);

    /* Load UNet and prepare inputs */
    MLModel *unet = load_model(@"unet", &err);
    if (!unet) NSLog(@"Can't load unet: %@\n", err);
    NSLog(@"Finished loading unet\n");

    float step = 1;
    shape = @[ @2, @4, @64, @64 ], strides = @[ @32768, @16384, @4096, @64 ];
    MLMultiArray *unet_latents = [[MLMultiArray alloc] initWithDataPointer:latents
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLMultiArray *unet_step = [[MLMultiArray alloc] initWithDataPointer:&step
        shape:@[ @1 ] dataType:MLMultiArrayDataTypeFloat32 strides:@[ @1 ]
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLMultiArray *unet_embeds = [MLMultiArray
        multiArrayByConcatenatingMultiArrays:@[ uncond, embeds ]
        alongAxis:0 dataType:MLMultiArrayDataTypeFloat32];
    MLDictionaryFeatureProvider* unet_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{ @"sample_1":unet_latents, @"timestep":unet_step, @"input_35":unet_embeds }
        error:&err];

    for (int i = 0; i < n_steps; ++i, step += 1000 / n_steps) {
        NSLog(@"Running step %d : %f\n", i, alphas[i + 1]);
        /* run UNet to predict noise residual */
        memcpy(latents[1], latents[0], n_noise * sizeof(float));
        id<MLFeatureProvider> unet_out = [unet predictionFromFeatures:unet_in error:&err];
        MLMultiArray *pred = [[unet_out featureValueForName:@"var_5609"] multiArrayValue];

        /* Perform guidance */
        float (*e_t)[n_noise] = pred.dataPointer;
        for (int i = 0; i < n_noise; ++i)
            e_t[0][i] += 7.5 * (e_t[1][i] - e_t[0][i]);

        /* Compute previous noisy sample */
        sched_step(alphas, i, e_t[0], latents[0], n_noise);
    }

    for (int i = 0; i < n_noise; ++i)
        latents[0][i] = 1.0 / 0.18215 * latents[0][i];

    /* Load VAE and prepare inputs */
    MLModel *post_quant = load_model(@"post_quant_conv", &err);
    if (!post_quant) NSLog(@"Can't load post_quant: %@\n", err);
    MLModel *decoder = load_model(@"vae_decoder", &err);
    if (!encoder) NSLog(@"Can't load decoder: %@\n", err);

    shape = @[ @1, @4, @64, @64 ], strides = @[ @16384, @16384, @4096, @64 ];
    MLMultiArray *quant_latents = [[MLMultiArray alloc] initWithDataPointer:latents[0]
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:^(void *bytes) { (void)bytes; } error:&err];
    MLDictionaryFeatureProvider* post_quant_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"input":quant_latents } error:&err];

    /* Decode latents into image */
    id<MLFeatureProvider> post_quant_out = [post_quant predictionFromFeatures:post_quant_in error:&err];
    MLMultiArray *z = [[post_quant_out featureValueForName:@"var_22"] multiArrayValue];
    MLDictionaryFeatureProvider *decoder_in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"z":z } error:&err];

    id<MLFeatureProvider> decoder_out = [decoder predictionFromFeatures:decoder_in error:&err];
    MLMultiArray *decoder_z = [[decoder_out featureValueForName:@"var_730"] multiArrayValue];
    float *decoded = decoder_z.dataPointer;

    /* Write out png with stb_image */
    unsigned char *image = calloc(512 * 512 * 3, sizeof(char));
    for (int i = 0; i < 512 * 512; ++i) {
        for (int j = 0; j < 3; ++j) {
            float pixel =  decoded[j * 512 * 512 + i];
            image[i * 3 + j] = (unsigned char)(pixel * 128 + 128);
        }
    }
    stbi_write_png("ruins_c.png", 512, 512, 3, image, 512 * 3);

    bpe_free(tokenizer);
    return 0;
}
