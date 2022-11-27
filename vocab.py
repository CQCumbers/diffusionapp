import json, re, struct
import requests

N_SLOTS = 65536
N_TOKENS = 4096
slot = struct.Struct("@IIi")

def hash_string(string):
    value = 2166136261
    for byte in bytes(string, "utf-8"):
        value = (value ^ byte) * 16777619
        value = value & 0xFFFFFFFF
    return value

def hash_mix(lhs, rhs):
    rotation = (lhs << 6) + (lhs >> 2)
    lhs ^= rhs + 0x9e3779b9 + rotation
    return lhs & 0xFFFFFFFF

def hash_add(buffer, key, value):
    offset, dist = key % N_SLOTS, 1
    while True:
        item = slot.unpack_from(buffer, offset * slot.size)
        if item[1] == key: return False
        slot.pack_into(buffer, offset * slot.size, dist, key, value)
        if item[0] == 0: return True
        if item[0] < dist: dist, key, value = item
        offset, dist = (offset + 1) % N_SLOTS, dist + 1

def read_rules(file):
    rules = bytearray(slot.size * N_SLOTS)
    for i, line in enumerate(file):
        if line.startswith(b"#"): continue
        lhs, rhs = line.decode("utf-8").split()
        rhs = rhs.replace("</w>", " ")
        key = hash_mix(hash_string(lhs), hash_string(rhs))
        hash_add(rules, key, i)
    return rules

def read_vocab(file):
    vocab, bos, eos = bytearray(slot.size * N_SLOTS), 0, 0
    for token, i in json.load(file).items():
        token = token.replace("</w>", " ")
        if token == "<|startoftext|>": bos = i
        elif token == "<|endoftext|>": eos = i
        else: hash_add(vocab, hash_string(token), i)
    return vocab, bos, eos

def convert_vocab(dst_path):
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/raw/main/tokenizer"
    resp = requests.get(f"{url}/merges.txt", stream=True)
    rules = read_rules(resp.raw)
    resp = requests.get(f"{url}/vocab.json", stream=True)
    vocab, bos, eos = read_vocab(resp.raw)

    with open(f"{dst_path}/bpe_context.bin", "wb+") as file:
        file.write(rules)
        file.write(vocab)
        file.write(bytearray(struct.calcsize("@hhhh") * N_TOKENS))
        file.write(bytearray(struct.calcsize("@ihh") * N_TOKENS * 4))
        file.write(struct.pack("@II", bos, eos))
