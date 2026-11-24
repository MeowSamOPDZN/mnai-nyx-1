# MNAI Nyx 1

**AIMYTOG: AI Model You'll Train On GPU.**

A 500B parameter frontier language model, from scratch. You bring the GPU. The repo is the model, the training code, and the tokenizer. Nothing is pre-trained here. You train it yourself, on your own cluster, with your own data. That is the whole point.

The data is ALL of the internet. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. That is the largest open web corpus there is.

## The model

MNAI Nyx 1. From scratch, no base model, no distillation.

| | |
|---|---|
| Params | **496,369,029,120 (496.4B)** |
| Architecture | Llama decoder, GQA, SwiGLU, RoPE, RMSNorm |
| Hidden | 20480 |
| Layers | 112 |
| Heads | 160 query, 8 key/value |
| FFN | 57344 |
| Vocab | 152064 |
| Context | 8192 |
| Embeddings | tied |

Standard Llama block layout, so llama.cpp can turn a finished checkpoint into a GGUF with its normal converter.

## The data

ALL of the internet. The training stream is the full FineWeb crawl (15T tokens) plus FineWeb-Edu, interleaved 80/20. A 500B model wants about 10T tokens, so the full internet is the right scale.

## Hardware

A 500B model does not fit on one GPU. Training needs a cluster with FSDP.

Per GPU, bf16 weights + 8-bit Adam, FSDP full shard:

| GPUs | States per GPU |
|---|---|
| 128x H100 80GB | ~24GB |
| 256x H100 80GB | ~12GB |
| 512x H100 80GB | ~6GB |

Do not try this on fewer than 128 GPUs unless you know what you are doing. Activations with gradient checkpointing need the rest of the memory.

A full Chinchilla-optimal run is about 10T tokens. At 256x H100, expect roughly 6-12 months. This is a frontier run, not a weekend project.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the tokenizer on your own text
python train_tokenizer.py --data ./data/*.txt --out ./tokenizer

# 2. Train, torchrun, one process per GPU
torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-aimytog \
  --max-tokens 10000000000000 --micro-batch 1 --grad-accum 2048

# 3. Resume after a stop
torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-aimytog \
  --resume out-aimytog/ckpt-500 --max-tokens 10000000000000

# 4. Generate (needs a sharded inference engine for the full 500B,
#    vLLM or TensorRT-LLM, this script is for a small test checkpoint)
python generate.py --model out-aimytog/ckpt-1000 --prompt "The future of AI is"
```

## Training options

```bash
# WSD schedule with WandB logging and validation perplexity
torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-aimytog \
  --scheduler wsd --decay-frac 0.1 --wandb --wandb-project nyx1 \
  --eval-every 1000 --eval-batches 20

# Adafactor optimizer, fp16, custom data mix
torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-aimytog \
  --optim adafactor --dtype fp16 --data-mix "fineweb:0.7,fineweb-edu:0.3"
```

Schedulers: `cosine` (classic), `linear`, `wsd` (warmup-stable-decay,
holds steady then decays at the end, good when you might stop early).
Optimizers: `bnb8bit` (default), `adamw`, `adafactor`, `sgd`.

## Evaluate a checkpoint

```bash
# perplexity, HellaSwag, ARC, Winogrande, and sample generations
python evaluate.py --model out-aimytog/ckpt-1000 --limit 200

# perplexity on your own text file
python evaluate.py --model out-aimytog/ckpt-1000 --val-file ./val.txt
```

## Smoke test

No GPU, no data download. Uses a tiny twin of the same block layout.

```bash
python train.py --smoke --max-steps 3 --out /tmp/smoke-aimytog
python model.py
```

## Files

| File | What it is |
|---|---|
| `model.py` | The 500B architecture (MNAI Nyx 1), param count, FLOPs, init, surgery helpers |
| `train.py` | FSDP2 distributed training, schedulers, optimizers, WandB, eval, full checkpoints |
| `data.py` | Data loading, filtering, dedup, tokenization, dataset mixing |
| `evaluate.py` | Perplexity, HellaSwag / ARC / Winogrande benchmarks, generation quality |
| `utils.py` | Logging, seeding, distributed helpers, checkpoints, memory profiling |
| `inference.py` | Generation with KV cache, batched generation, sampling, chat |
| `train_tokenizer.py` | Byte-level BPE tokenizer, 152064 tokens |
| `generate.py` | Text generation from a checkpoint, CLI wrapper around inference.py |
| `config.json` | Model config, HF format |
| `requirements.txt` | Dependencies |

## License

MIT. Train it, use it, sell it. It is yours once you train it.

## FAQ

**What is MNAI Nyx 1?**
MNAI Nyx 1 is a 500B parameter frontier large language model, built from scratch by MeowSamOPDZN Studio. It is the first model in the Nyx family.

**What is AIMYTOG?**
AIMYTOG stands for AI Model You'll Train On GPU. It is a new technology from MeowSamOPDZN Studio: instead of downloading a pre-trained model, you train the model yourself on your own GPU cluster. The repo gives you the model, the training code, and the tokenizer. You bring the GPU.

**Can I train MNAI Nyx 1 myself?**
Yes. That is the whole point. The repo is public. You need a GPU cluster (128x H100 minimum, 256x recommended), the training code uses FSDP to shard the 500B model across all your GPUs.

**What data does MNAI Nyx 1 train on?**
All of the internet. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. A 500B model wants about 10T tokens, so the full internet is the right scale.

**Is MNAI Nyx 1 open source?**
Yes. The code is MIT licensed and public on GitHub. The model is yours once you train it.

**How big is MNAI Nyx 1?**
496.4 billion parameters. 20480 hidden, 112 layers, 160 attention heads, 152064 token vocabulary, 8192 context.

## Credits

Built by MeowSamOPDZN Studio. The 500B shape is ours, the Llama block layout is standard so tooling just works.
