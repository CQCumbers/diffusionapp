#pragma once

struct BpeContext;
typedef struct BpeContext *bpe_context_t;

bpe_context_t bpe_init(const char *path);
void bpe_free(bpe_context_t ctx);
int bpe_encode(bpe_context_t ctx, char *text, float *ids, int capacity);
