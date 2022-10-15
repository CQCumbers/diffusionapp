#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int rule;
    short token_l;
    short byte_r;
} pair_t;

typedef struct {
    short prev, next;
    short byte_l, byte_r;
} token_t;

typedef struct {
    unsigned dist;
    unsigned hash;
    int rule;
} slot_t;

#define N_SLOTS 65536
#define N_TOKENS 4096

struct BpeContext {
    slot_t rules[N_SLOTS];
    slot_t vocab[N_SLOTS];
    token_t list[N_TOKENS];
    pair_t agenda[N_TOKENS * 4];
    unsigned bos_id, eos_id;
};

/* === Data structures === */

static void heap_pop(pair_t *data, unsigned size) {
    unsigned index = 0, child = 0;
    pair_t value = data[size - 1];
    while ((child = index * 2 + 1) < size) {
        child += child + 1 < size && data[child + 1].rule < data[child].rule;
        if (value.rule < data[child].rule) break;
        data[index] = data[child], index = child;
    }
    data[index] = value;
}

static void heap_push(pair_t *data, unsigned size, pair_t value) {
    while (size > 0) {
        unsigned parent = (size - 1) / 2;
        if (data[parent].rule < value.rule) break;
        data[size] = data[parent], size = parent;
    }
    data[size] = value;
}

static void heap_make(pair_t *data, unsigned size) {
    if (size <= 1) return;
    for (unsigned item = (size - 2) / 2;; --item) {
        pair_t value = data[item];
        unsigned index = item, child = 0;
        while ((child = index * 2 + 1) < size) {
            child += child + 1 < size && data[child + 1].rule < data[child].rule;
            if (value.rule < data[child].rule) break;
            data[index] = data[child], index = child;
        }
        data[index] = value;
        if (item == 0) break;
    }
}

static int hash_add(slot_t *table, unsigned size, unsigned hash, int rule) {
    slot_t item = { 1, hash, rule }, tmp;
    for (unsigned i = hash;; ++item.dist, ++i) {
        slot_t *slot = &table[i & (size - 1)];
        if (slot->dist == 0) return *slot = item, 1;
        if (slot->dist < item.dist) tmp = item, item = *slot, *slot = tmp;
        if (slot->hash == item.hash) return 0;
    }
}

static int hash_get(const slot_t *table, unsigned size, unsigned hash) {
    if (size == 0) return -1;
    for (unsigned dist = 1, i = hash;; ++dist, ++i) {
        const slot_t *slot = &table[i & (size - 1)];
        if (slot->dist < dist) return -1;
        else if (slot->hash == hash) return slot->rule;
    }
}

static unsigned hash_string(const char *string) {
    unsigned hash = 2166136261;
    for (; *string != '\0'; ++string)
        hash = (hash ^ *string) * 16777619;
    return hash;
}

static unsigned hash_token(token_t token, const char *text) {
    unsigned hash = 2166136261;
    const char *ptr = text + token.byte_l;
    for (; ptr != text + token.byte_r; ++ptr)
        hash = (hash ^ *ptr) * 16777619;
    return hash;
}

static unsigned hash_mix(unsigned lhs, unsigned rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

/* === Tokenization helpers === */

static int rule_get(const slot_t *rules,
        token_t token_l, token_t token_r, const char *text) {
    if (text[token_r.byte_l] == ' ') return 0;
    unsigned hash_l = hash_token(token_l, text);
    unsigned hash_r = hash_token(token_r, text);
    int val = hash_get(rules, N_SLOTS, hash_mix(hash_l, hash_r));
    return val;
}

static char *replace_end(char *string) {
    /* replace word terminator with space */
    size_t slen = strlen(string);
    if (slen < 4 || strcmp(string + slen - 4, "</w>")) return string;
    string[slen - 4] = ' ', string[slen - 3] = '\0';
    return string;
}


#define MAX_LINE 512

static void rule_init(bpe_context_t ctx, FILE *file) {
    char buf[MAX_LINE];
    for (int i = 0; fgets(buf, MAX_LINE, file); ++i) {
        if (i * 8 > N_SLOTS * 7)
            return (void)printf("Max rules reached: %d\n", i);
        if (i == 0) continue;

        char *token_l = strtok(buf, " \n");
        char *token_r = replace_end(strtok(NULL, " \n"));
        unsigned hash_l = hash_string(token_l);
        unsigned hash_r = hash_string(token_r);
        hash_add(ctx->rules, N_SLOTS, hash_mix(hash_l, hash_r), i);
    }
}

static void vocab_init(bpe_context_t ctx, FILE *file) {
    char string[MAX_LINE], number[MAX_LINE];
    char ch, *end;
    int n_string, n_number;
    int state = 0, token_id = 0;

    unsigned bos_hash = hash_string("<|startoftext|>");
    unsigned eos_hash = hash_string("<|endoftext|>");
    unsigned hash;

    while (fread(&ch, 1, 1, file)) {
        switch (state) {
        case 0:
            if (ch == '{') state = 1;
            break;
        case 1:
            if (ch == '"') state = 2;
            n_string = 0;
            break;
        case 2:
            if (ch == '\\') state = 3;
            else if (ch == '"') state = 4;
            else string[n_string++] = ch;
            break;
        case 3:
            string[n_string++] = ch;
            state = 2;
            break;
        case 4:
           if (ch == ':') state = 5;
           n_number = 0;
           break; 
        case 5:
           if (ch == ',') state = 1;
           if (ch == '}') state = 0;
           if (state != 5) {
               string[n_string] = '\0';
               hash = hash_string(replace_end(string));
               end = number + n_number;
               token_id = strtol(number, &end, 10);

               if (hash == bos_hash) { ctx->bos_id = token_id; break; }
               if (hash == eos_hash) { ctx->eos_id = token_id; break; }
               hash_add(ctx->vocab, N_SLOTS, hash, token_id);
           } else number[n_number++] = ch;
           break;
        }
    }
}

/* === Interface functions === */

bpe_context_t bpe_init(const char *vocab, const char *merges) {
    bpe_context_t ctx = calloc(1, sizeof(struct BpeContext));
    FILE *vocab_file = fopen(vocab, "r");
    if (!vocab_file) return printf("No vocab: %s\n", vocab), ctx;
    vocab_init(ctx, vocab_file);
    fclose(vocab_file);

    FILE *merges_file = fopen(merges, "r");
    if (!merges_file) return printf("No merges: %s\n", merges), ctx;
    rule_init(ctx, merges_file);
    fclose(merges_file);
    return ctx;
}

void bpe_free(bpe_context_t ctx) {
    free(ctx);
}

int bpe_encode(bpe_context_t ctx, const char *text, float *ids, int capacity) {
    /* initialize token list */
    token_t *list = ctx->list;
    short length = strlen(text), n_ids = 0;
    if (length >= N_TOKENS) return printf("Too much text%d\n", length), 0;
    for (short i = 0; i < length; ++i)
        list[i] = (token_t){ i - 1, i + 1, i, i + 1 };
    if (length) list[length - 1].next = -1;

    /* initialize merge agenda */
    unsigned n_agenda = 0;
    for (short i = 1; i < length; ++i) {
        int rule = rule_get(ctx->rules, list[i - 1], list[i], text);
        pair_t new_pair = { rule, i - 1, list[i].byte_r };
        if (rule != -1) ctx->agenda[n_agenda++] = new_pair;
    }
    heap_make(ctx->agenda, n_agenda);

    /* process merges in agenda */
    while (n_agenda > 0) {
        pair_t pair = ctx->agenda[0];
        heap_pop(ctx->agenda, n_agenda--);

        /* check pair still valid */
        short curr = pair.token_l;
        short skip = list[curr].next;
        if (skip == -1 || list[skip].byte_r != pair.byte_r) continue;

        /* merge tokens in linked list */
        short prev = list[curr].prev;
        short next = list[skip].next;
        if (next != -1) list[next].prev = curr;
        list[curr].byte_r = list[skip].byte_r;
        list[curr].next = next;
        list[skip] = (token_t){ -1, -1, -1, -1 };

        /* add new pairs to agenda */
        if (prev != -1) {
            int rule = rule_get(ctx->rules, list[prev], list[curr], text);
            pair_t new_pair = { rule, prev, list[curr].byte_r };
            if (rule != -1) heap_push(ctx->agenda, n_agenda++, new_pair);
        }
        if (next != -1) {
            int rule = rule_get(ctx->rules, list[curr], list[next], text);
            pair_t new_pair = { rule, curr, list[next].byte_r };
            if (rule != -1) heap_push(ctx->agenda, n_agenda++, new_pair);
        }
    }

    /* convert merged tokens to ids */
    ids[n_ids] = ctx->bos_id;
    for (int i = 0; i < length; ++i) {
        if (list[i].byte_l == -1) continue;
        unsigned hash = hash_token(list[i], text);
        int token_id = hash_get(ctx->vocab, N_SLOTS, hash);
        if (token_id == -1) token_id = ctx->eos_id;
        if (++n_ids < capacity) ids[n_ids] = token_id;
    }

    for (int i = n_ids + 1; i < capacity; ++i)
        ids[i] = ctx->eos_id;
    return n_ids + 1;
}
