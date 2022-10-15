#pragma once

#define N_TOKENS 4096

typedef enum {
    T2I_UNLOADED,
    T2I_ENCODER_LOADED,
    T2I_ENCODER_NOLOAD,
    T2I_ENCODER_FAILED,
    T2I_UNET_LOADED,
    T2I_UNET_NOLOAD,
    T2I_UNET_FAILED,
    T2I_DECODER_LOADED,
    T2I_DECODER_NOLOAD,
    T2I_DECODER_FAILED,
    T2I_FINISHED,
    T2I_STEPS,
} t2i_status_t;

typedef struct {
    char prompt[N_TOKENS];
    char image[512 * 512 * 3];
    int strength, steps;
    int guide, seed;
} t2i_request_t;

struct T2IEngine;
typedef struct T2IEngine *t2i_t;
typedef int (*t2i_handler_t)(void *ctx, int req_id, int status);

t2i_t t2i_init(t2i_handler_t handler, void *ctx);
void t2i_free(t2i_t engine);
t2i_request_t *t2i_request(t2i_t engine, int req_id);
int t2i_acquire(t2i_t engine);
void t2i_submit(t2i_t engine, int req_id);
