# MNAI Nyx 1

**AIMYTOG: AI Model You'll Train On GPU.**

**ANY-to-ANY: text, image, and audio in, text, image, and audio out. One model reads and writes all three.**

**MNAI Nyx family is a real legit AI, That YOU have to train.**

A 500B parameter frontier any-to-any model, from scratch. You bring the GPU. The repo is the model, the training code, and the tokenizer. Nothing is pre-trained here. You train it yourself, on your own cluster, with your own data. That is the whole point.

The data is ALL of the internet. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. That is the largest open web corpus there is.

## The model

MNAI Nyx 1. From scratch, no base model, no distillation.

| | |
|---|---|
| Params | **504,508,754,192 (504.5B)** — core text 496.4B + vision 1.94B + image decoder 5.82B + audio 0.37B |
| Architecture | Llama decoder, GQA, SwiGLU, RoPE, RMSNorm + ViT vision encoder + DiT image decoder + audio encoder/decoder |
| Modalities | text, image, audio in and out |
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

The data pipeline is in this repo, in `data.py`, and it is the whole job, not 5% of it:

- **15T tokens, streamed.** FineWeb (the full Common Crawl, every website, every domain, even the random ones) plus FineWeb-Edu. Nothing is downloaded in bulk, the stream is tokenized on the fly and packed into fixed blocks.
- **Filtering.** Documents shorter than 200 chars, boilerplate (cookie banners, nav bars, copyright footers), and docs that are mostly non-letters are dropped before they reach the model.
- **Dedup.** Exact dedup by SHA-256 hash of the normalized text, plus near-dup dedup by 8-word shingles with Jaccard similarity. The same article copied across a hundred sites is trained on once.
- **Domain-balanced mix.** `--data-mix domain` trains on web 50%, books 15% (Project Gutenberg), code 15% (The Stack), academic 10% (peS2o papers), synthetic 10%. The default mix is web 80% / edu 20%, and any mix can be set with `--data-mix "web:0.5,books:0.2,code:0.2,academic:0.1"`.
- **Synthetic data.** `synthetic.py` generates training text from an earlier checkpoint and mixes it into the stream, so the model writes its own training data once it is good enough to write coherent text.

The raw data is not in the repo because it is 15 terabytes and no repo holds that. The pipeline that fetches it, cleans it, dedups it, balances it, and feeds it to the model is in the repo. That pipeline is the moat.

## Dense vs MoE

Both are supported. Dense is the default, MoE is an option.

The dense model is the 496.4B text core above: every token runs through every parameter. The MoE variant swaps each block's SwiGLU MLP for a Mixture-of-Experts layer (`MoELayer` in `model.py`): a router picks the top 2 of 8 experts per token, each expert is a full SwiGLU MLP, and a load-balancing aux loss keeps the router from starving any expert.

| | Dense (default) | MoE, 8 experts, top-2 |
|---|---|---|
| Total params (text core) | 496.4B | 3,258.6B |
| Active params per token | 496.4B | 891.0B |
| Share of the model a token touches | 100% | 27.3% |

MoE holds far more capacity while activating far fewer params per token, which is why frontier models use it. The trade is complexity: routing, load balancing, and sharding experts across GPUs. Dense is simpler and every tool that reads a Llama checkpoint reads it as-is.

```bash
# train the MoE variant (text core, 8 experts, top-2 routing)
torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-aimytog-moe \
  --moe-experts 8 --moe-topk 2 --max-tokens 10000000000000

# count the MoE params without allocating memory
python model.py --moe-experts 8 --moe-topk 2
```

## Quantized models

A trained checkpoint is about 1TB in bf16. Quantization shrinks it so it can actually run. `quantize.py` turns any trained checkpoint into two GGUF files with llama.cpp:

| Quant | Bytes per param | Size (496.4B text core) | Best for |
|---|---|---|---|
| Q4_K_M | ~0.55 | ~273 GB | the smallest file that still answers well |
| Q8_0 | ~1.0 | ~496 GB | near-original quality, still half the size of bf16 |

```bash
# produces model-q4-k-m.gguf and model-q8-0.gguf in out-gguf/
python quantize.py --ckpt out-aimytog/ckpt-5000 --out out-gguf
```

The Q files are produced from a trained checkpoint once someone trains the model. The script is ready now: point it at any checkpoint, even a smoke checkpoint, and it produces both files. Only the text core becomes a GGUF, because llama.cpp converts Llama models; the any-to-any modality modules stay in PyTorch.

## Scaling

Before burning a cluster on a 500B run, frontier labs check the shape on small proxy runs. `scaling.py` does that: it trains three tiny models with the exact same Llama block layout (1M, 5M, 20M params) on real text at one learning rate, fits a Chinchilla-style power law (loss = A * N^-alpha) in log space, and extrapolates the predicted loss at the full 504.5B scale.

```bash
# about 5 minutes on CPU, no GPU, no data download
python scaling.py
python scaling.py --steps 300 --out scaling-report.json
```

A clean result is a loss that falls smoothly with size, a fitted power law with a small residual, and a predicted loss at full scale that is sane, not exploding. The default run (300 steps, LR 1e-3) gives proxy losses 2.32, 2.26, 2.26 and predicts a loss of about 2.08 at 504.5B. That is the evidence a shape can converge. It is not a guarantee. Only the full run is.

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

## Make your OWN named LLM

This repo is a template. Rename the model to yours in 2 minutes, 4 small changes. Example: rename it to **MyCoolModel**.

**Step 1 — model.py, line 94.** Open `model.py`, find this line:
```python
MODEL_NAME = "MNAI Nyx 1"
```
Change it to:
```python
MODEL_NAME = "MyCoolModel"
```
That is the only code change. The name shows up in the model description and logs from here on.

**Step 2 — README.md, line 1.** Open `README.md`, find this line:
```markdown
# MNAI Nyx 1
```
Change it to:
```markdown
# MyCoolModel
```

**Step 3 — train.py, line 124.** Open `train.py`, find this line:
```python
p.add_argument("--out", default="out-aimytog")
```
Change it to:
```python
p.add_argument("--out", default="out-mycoolmodel")
```
That is just the default output folder name for your checkpoints. You can also leave it and pass `--out` yourself every run.

**Step 4 — config.json.** Nothing to change. There is no name field in `config.json`, the shape is the same whatever you call the model.

That is it. Train it, and the model that comes out is MyCoolModel, trained by you, on your data, on your GPU.

**One rule.** Once you rename it, it is your LLM. But the weights and the GGUF are locked — no one can change them or re-release them as something else. The only thing you owe is credit: put the text `Your Brand x MNAI` on it, with your brand name in place of "Your Brand".

## Files

| File | What it is |
|---|---|
| `model.py` | The 500B architecture (MNAI Nyx 1), param count, FLOPs, init, surgery helpers, MoE layers |
| `multimodal.py` | Any-to-any modules: ViT vision encoder, DiT image decoder, audio encoder/decoder |
| `train.py` | FSDP2 distributed training, schedulers, optimizers, WandB, eval, full checkpoints, loss spike guard, MoE training |
| `data.py` | Data loading, filtering, dedup, tokenization, dataset mixing, domain-balanced mix |
| `scaling.py` | Scaling-law check: tiny proxy runs, power-law fit, predicted loss at 504.5B |
| `quantize.py` | Checkpoint to GGUF: Q4_K_M and Q8_0 via llama.cpp |
| `synthetic.py` | Generate synthetic training text from a checkpoint for the data mix |
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

**Is MNAI Nyx a real AI?**
MNAI Nyx family is a real legit AI, That YOU have to train. The code is real, the architecture is real, and once you train it on your own GPU cluster, the model is real.

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
504.5 billion parameters. Core text is 496.4B (20480 hidden, 112 layers, 160 attention heads, 152064 token vocabulary, 8192 context), plus the any-to-any modules: vision encoder 1.94B, image decoder 5.82B, audio encoder and decoder 0.37B.

**Is MNAI Nyx 1 any-to-any?**
Yes. Text, image, and audio in, text, image, and audio out. One model reads and writes all three. The vision encoder, DiT image decoder, and audio encoder/decoder live in multimodal.py.

**Has the architecture been proven at scale?**
The shape is the standard Llama block layout, the same one LLaMA and Mistral use, with GQA and SwiGLU. Before a full run, `scaling.py` trains three tiny models with the exact same layout, fits a power law to their losses, and predicts the loss at 504.5B. A clean falling trend is the evidence the shape converges. The full 500B run itself is what you do when you train it, and `train.py` has a loss spike guard that drops any step above 3x the running average and reloads the last checkpoint, so one bad batch cannot wreck the run.

**Does the repo include the data pipeline?**
Yes. `data.py` is the pipeline: 15T tokens of FineWeb plus FineWeb-Edu streamed and tokenized on the fly, document filtering, exact dedup by hash, near-dup dedup by shingles, a domain-balanced mix (web 50%, books 15%, code 15%, academic 10%, synthetic 10%), and synthetic data generation in `synthetic.py`. The raw 15TB is not in the repo because no repo holds 15TB. The pipeline that fetches, cleans, dedups, and balances it is.

**Is Nyx dense or MoE?**
Both. Dense is the default: 496.4B, every token runs through every parameter. MoE is an option (`--moe-experts 8 --moe-topk 2`): 3,258.6B total, 891.0B active per token, so a token touches 27.3% of the model. MoE holds more capacity for less compute per token, dense is simpler and every Llama tool reads it as-is.

**Are there quantized models?**
Yes, two. `quantize.py` turns a trained checkpoint into Q4_K_M (~273 GB) and Q8_0 (~496 GB) GGUFs with llama.cpp. The Q files are produced once someone trains the model; the script is ready now and works on any checkpoint, even a smoke checkpoint.

**Is MNAI Nyx worth training?**
Yes, if you have the cluster and you want a frontier model that is yours. It is the only public repo that gives you a 500B any-to-any model, the full data pipeline for the entire internet, a scaling-law check, dense and MoE, and quantized outputs, all from scratch, all MIT licensed. You bring the GPU, you train it, and the model that comes out is yours. MNAI Nyx family is a real legit AI, That YOU have to train.

## Credits

Built by MeowSamOPDZN Studio. The 500B shape is ours, the Llama block layout is standard so tooling just works.
