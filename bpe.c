#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned rule;
    short token_l, token_r;
} pair_t;

void print_pairs(pair_t *pairs, int num) {
    for (pair_t *it = pairs; it < pairs + num; ++it) {
        printf("Pair rule %u: %hd <-> %hd\n", it->rule, it->token_l, it->token_r);
    }
}

typedef struct {
    short prev, next;
    short byte_l, byte_r;
} token_t;

void print_tokens(token_t *tokens, int num) {
    for (token_t *it = tokens; it < tokens + num; ++it) {
        printf("Token %hd <-> %hd, prev %hd, next %hd\n",
            it->byte_l, it->byte_r, it->prev, it->next);
    }
}

void heap_pop(pair_t *data, unsigned size) {
    int index = 0, child = 0;
    pair_t value = data[size - 1];
    while ((child = index * 2 + 1) < size) {
        child += child + 1 < size && data[child + 1].rule < data[child].rule;
        if (value.rule < data[child].rule) break;
        data[index] = data[child], index = child;
    }
    data[index] = value;
}

void heap_push(pair_t *data, unsigned size, pair_t value) {
    while (size > 0) {
        int parent = (size - 1) / 2;
        if (data[parent].rule < value.rule) break;
        data[size] = data[parent], size = parent;
    }
    data[size] = value;
}

void heap_make(pair_t *data, unsigned size) {
    if (size <= 1) return;
    for (int item = (size - 2) / 2;; --item) {
        pair_t value = data[item];
        int index = item, child = 0;
        while ((child = index * 2 + 1) < size) {
            child += child + 1 < size && data[child + 1].rule < data[child].rule;
            if (value.rule < data[child].rule) break;
            data[index] = data[child], index = child;
        }
        data[index] = value;
        if (item == 0) break;
    }
}

typedef struct {
    int dist;
    unsigned hash;
    unsigned rule;
} slot_t;

unsigned hash_add(slot_t *table, unsigned size, unsigned hash, unsigned rule) {
    slot_t item = { 1, hash, rule }, tmp;
    for (unsigned i = hash;; ++item.dist, ++i) {
        slot_t *slot = &table[i & (size - 1)];
        if (slot->dist == 0) return *slot = item, 1;
        if (slot->dist < item.dist) tmp = item, item = *slot, *slot = tmp;
        if (slot->hash == item.hash) return 0;
    }
}

unsigned hash_get(const slot_t *table, unsigned size, unsigned hash) {
    if (size == 0) return -1;
    for (unsigned dist = 1, i = hash;; ++dist, ++i) {
        const slot_t *slot = &table[i & (size - 1)];
        if (slot->dist < dist) return -1;
        else if (slot->hash == hash) return slot->rule;
    }
}

unsigned hash(const char *start, const char *end) {
    /* 32-bit FNV-1a */
    unsigned hash = 2166136261;
    for (; start != end; ++start)
        hash = (hash ^ *start) * 16777619;
    return hash;
}

unsigned rule_get(const slot_t *rules, unsigned n_rules,
        token_t token_l, token_t token_r, const char *text) {
    if (text[token_r.byte_l] == ' ') return 0;
    /* store output token hash with rule? */
    unsigned hash_l = hash(text + token_l.byte_l, text + token_l.byte_r);
    unsigned hash_r = hash(text + token_r.byte_l, text + token_r.byte_r);
    hash_l ^= hash_r + 0x9e3779b9 + (hash_l << 6) + (hash_l >> 2);
    printf("Hash for %.*s THEN %.*s: %x\n",
        token_l.byte_r - token_l.byte_l, text + token_l.byte_l,
        token_r.byte_r - token_r.byte_l, text + token_r.byte_l,
        hash_l);
    unsigned val = hash_get(rules, n_rules, hash_l);
    printf("Got rule %u\n", val);
    return val;
}

#define MAX_LIST 100
#define MAX_AGENDA 1000
#define MAX_RULES 65536
#define MAX_LINE 300

token_t list[MAX_LIST];
pair_t agenda[MAX_AGENDA];
slot_t rules[MAX_RULES];

slot_t vocab[MAX_RULES];

char *replace_end(char *string) {
    size_t slen = strlen(string);
    if (slen < 4 || strcmp(string + slen - 4, "</w>")) return string;
    string[slen - 4] = ' ', string[slen - 3] = '\0';
    return string;
}

void read_rules(FILE *file) {
    char buf[MAX_LINE];
    for (int i = 0; fgets(buf, MAX_LINE, file); ++i) {
        if (i == 0) continue;
        char *token_l = strtok(buf, " \n");
        char *token_r = replace_end(strtok(NULL, " \n"));
        unsigned hash_l = hash(token_l, token_l + strlen(token_l));
        unsigned hash_r = hash(token_r, token_r + strlen(token_r));
        hash_l ^= hash_r + 0x9e3779b9 + (hash_l << 6) + (hash_l >> 2);
        hash_add(rules, MAX_RULES, hash_l, i);
    }
}

void read_vocab(FILE *file) {
    char string[MAX_LINE], number[MAX_LINE], ch, *end;
    int n_string, n_number, state = 0;
    unsigned hash_ = 0, token_id = 0;
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
               replace_end(string);
               hash_ = hash(string, string + strlen(string));
               end = number + n_number;
               token_id = strtol(number, &end, 10);
               printf("%d Adding %.*s with hash %x to vocab\n", token_id, n_string, string, hash_);
               hash_add(vocab, MAX_RULES, hash_, token_id);
           } else number[n_number++] = ch;
           break;
        }
    }
}


int main(int argc, char *argv[]) {
    /* TODO: avoid overflow conditions */
    if (argc < 2) return puts("Usage: ./tokenize <text>");
    const char *text = argv[1];
    short n_list = strlen(argv[1]);
    unsigned n_rules = MAX_RULES;
    
    /* initialize rule table */
    FILE *merges_file = fopen("merges.txt", "r");
    read_rules(merges_file);
    fclose(merges_file);

    FILE *vocab_file = fopen("vocab.json", "r");
    read_vocab(vocab_file);
    fclose(vocab_file);

    /*#define N_MERGES 7
    const char *merges[N_MERGES][2] = {
        { "e", "r" },
        { "h", "e" },
        { "l", "l" },
        { "l", "o" },
        { "he", "ll" },
        { "lo", "w" },
        { "hell", "o" }
    };

    for (unsigned i = 0; i < N_MERGES; ++i) {
        unsigned hash_l = hash(merges[i][0], merges[i][0] + strlen(merges[i][0]));
        unsigned hash_r = hash(merges[i][1], merges[i][1] + strlen(merges[i][1]));
        hash_l ^= hash_r + 0x9e3779b9 + (hash_l << 6) + (hash_l >> 2);
        printf("Adding hash %x for %s then %s: %d\n", hash_l, merges[i][0], merges[i][1], i);
        hash_add(rules, n_rules, hash_l, i);
    }*/

    /* initialize token list */
    puts("Initializing token list");
    for (short i = 0; i < n_list; ++i)
        list[i] = (token_t){ i - 1, i + 1, i, i + 1 };
    list[n_list - 1].next = -1;

    /* initialize merge agenda */
    puts("Initializing agenda");
    unsigned n_agenda = 0;
    for (short i = 1; i < n_list; ++i) {
        unsigned rule = rule_get(rules, n_rules, list[i - 1], list[i], text);
        pair_t new_pair = { rule, i - 1, i };
        if (rule != -1) agenda[n_agenda++] = new_pair;
    }
    heap_make(agenda, n_agenda);

    printf("n_agenda initially %d\n", n_agenda);
    print_pairs(agenda, n_agenda);
    print_tokens(list, n_list);

    while (n_agenda > 0) {
        pair_t pair = agenda[0];
        heap_pop(agenda, n_agenda--);

        /* check pair still valid */
        short curr = pair.token_l;
        short skip = pair.token_r;
        if (list[curr].next != skip) continue;

        /* merge tokens in linked list */
        short prev = list[curr].prev;
        short next = list[skip].next;
        if (next != -1) list[next].prev = curr;
        list[curr].byte_r = list[skip].byte_r;
        list[curr].next = next;
        list[skip] = (token_t){ -1, -1, -1, -1 };

        /* add new pairs to agenda */
        if (prev != -1) {
            unsigned rule = rule_get(rules, n_rules, list[prev], list[curr], text);
            pair_t new_pair = { rule, prev, curr };
            if (rule != -1) heap_push(agenda, n_agenda++, new_pair);
        }
        if (next != -1) {
            unsigned rule = rule_get(rules, n_rules, list[curr], list[next], text);
            pair_t new_pair = { rule, curr, next };
            if (rule != -1) heap_push(agenda, n_agenda++, new_pair);
        }

        printf("n_agenda now %d\n", n_agenda);
        print_pairs(agenda, n_agenda);
        print_tokens(list, n_list);
    }

    for (short i = 0; i < n_list; ++i) {
        if (list[i].byte_l == -1) continue;
        printf("'%.*s' ", list[i].byte_r - list[i].byte_l, text + list[i].byte_l);
    }
    printf("\n");

    for (int i = 0; i < n_list; ++i) {
        if (list[i].byte_l == -1) continue;
        unsigned hash_ = hash(text + list[i].byte_l, text + list[i].byte_r);
        printf("%d ", hash_get(vocab, MAX_RULES, hash_));
    }
    printf("\n");

    return 0;
}
