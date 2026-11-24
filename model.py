"""MNAI Nyx 1, the 500B model. AIMYTOG: AI Model You'll Train On GPU.

You bring the GPU. The repo is the model, the training code, and the
tokenizer. Nothing is pre-trained here. You train it yourself, on your
own cluster, with your own data. That is the whole point.

The data is ALL of the internet. The training stream is the full
FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu.
That is the largest open web corpus there is. A 500B model wants
about 10T tokens, so the full internet is the right scale.

The shape is ours: 20480 hidden, 112 layers, 160 query heads with only
8 key/value heads, a 57344-wide SwiGLU feed-forward, RoPE positions,
RMSNorm, no biases, tied embeddings, 152064-token vocab, 8192 context.
It adds up to about 500B parameters.

Why this shape: the GQA ratio keeps attention cheap at this scale,
the wide SwiGLU carries most of the capacity, and tying the output
head to the input embeddings saves about 3B params for free. It is
built on the standard Llama block layout on purpose, so llama.cpp can
turn a finished checkpoint into a GGUF with its normal converter, no
custom converter needed.

A 500B model does not fit on one GPU. Training needs a cluster with
FSDP or DeepSpeed, 8-bit optimizer states, gradient checkpointing,
and bf16. The train.py in this repo wires all of that up. See the
README for the hardware math before you start.
"""
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# the shape, all in one place so train.py and generate.py agree
VOCAB_SIZE = 152064
HIDDEN_SIZE = 20480
NUM_LAYERS = 112
NUM_HEADS = 160
NUM_KV_HEADS = 8
INTERMEDIATE_SIZE = 57344
MAX_CTX = 8192
ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5


def build_config():
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_CTX,
        rope_theta=ROPE_THETA,
        rms_norm_eps=RMS_NORM_EPS,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def build_model(dtype=torch.bfloat16):
    cfg = build_config()
    model = LlamaForCausalLM(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model):
    total, _ = count_params(model)
    lines = [
        f"vocab {VOCAB_SIZE}, hidden {HIDDEN_SIZE}, layers {NUM_LAYERS}",
        f"heads {NUM_HEADS}, kv heads {NUM_KV_HEADS}, ffn {INTERMEDIATE_SIZE}",
        f"context {MAX_CTX}, rope theta {int(ROPE_THETA)}",
        f"params {total:,} ({total / 1e9:.1f}B), tied embeddings: yes",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # meta device: count params without allocating 1TB of RAM
    with torch.device("meta"):
        model = build_model(dtype=None)
    print(describe(model))
    print()
    print("this model is 500B params. Do not build it on one GPU.")
    print("train.py uses FSDP to shard it across a cluster.")
