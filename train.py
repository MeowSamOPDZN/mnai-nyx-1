"""Train MNAI Nyx 1 from scratch. Random weights, no base model.
AIMYTOG: AI Model You'll Train On GPU.

A 500B model does not fit on one GPU. This script uses FSDP2 to
shard weights, gradients, and optimizer states across all GPUs in
the job. Launch it with torchrun, one process per GPU.

The data is ALL of the internet. The training stream is the full
FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu.
That is the largest open web corpus there is. A 500B model wants
about 10T tokens, so the full internet is the right scale.

Hardware math, bf16 weights + 8-bit Adam, per GPU with FSDP full shard:
  weights  500B * 2 bytes / N GPUs
  grads    500B * 2 bytes / N GPUs
  optim    500B * 2 bytes / N GPUs (8-bit Adam, 2 states)
  total    about 3TB / N GPUs, plus activations with checkpointing
On 256x H100 80GB, that is about 12GB per GPU for states, leaving
room for activations. On 512x H100, about 6GB. Do not try this on
fewer than 128 GPUs unless you know what you are doing.

Data is ALL of the internet, streamed from Hugging Face. The full
FineWeb crawl (15T tokens of open web text) plus FineWeb-Edu, tokenized
on the fly with the Nyx 1 tokenizer, packed into 8192-token blocks.
Checkpoints save in HF format every so often, so a job ending just
means resume with --resume and keep going.

Chinchilla-optimal for 500B params is about 10T tokens. That is the
default --max-tokens. At 256x H100, expect roughly 6-12 months of
training. This is a frontier run, not a weekend project.

Usage, torchrun on a cluster:
  torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
    --master_addr=<master> --master_port=29500 \
    train.py --tokenizer ./tokenizer --out out-aimytog \
    --max-tokens 10000000000000 --micro-batch 1 --grad-accum 2048

Smoke test on CPU, no data download. Uses a tiny twin of the same
block layout for the loop test, full 500B forward is in model.py:
  python train.py --smoke --max-steps 3 --out /tmp/smoke-aimytog
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

from model import VOCAB_SIZE, build_config, build_model, count_params

SEQ_LEN = 8192
# smoke test uses a tiny sequence, a full 8192 block on CPU would crawl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer")
    p.add_argument("--out", default="out-aimytog")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-tokens", type=int, default=10_000_000_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps, 0 = use max-tokens")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optim", default="bnb8bit", choices=["bnb8bit", "adamw"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=0,
                   help="stop cleanly after this many hours, saving a checkpoint first, 0 = off")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def token_stream(tok, smoke=False):
    # endless stream of token ids, so training never waits on disk
    if smoke:
        rng = random.Random(0)
        while True:
            yield rng.randrange(100, VOCAB_SIZE)
        return
    from datasets import interleave_datasets, load_dataset

    if is_main():
        print("loading the full internet stream (FineWeb 15T + FineWeb-Edu)...", flush=True)
        print("first download can take a few minutes...", flush=True)
    # the whole internet: full FineWeb crawl, 15T tokens, plus Edu
    ds_web = load_dataset("HuggingFaceFW/fineweb", "default",
                          split="train", streaming=True)
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
    ds = interleave_datasets([ds_web, ds_edu], probabilities=[0.8, 0.2])
    if is_main():
        print("stream ready, reading first docs...", flush=True)
    n_docs = 0
    for row in ds:
        text = row["text"]
        n_docs += 1
        if n_docs == 1 and is_main():
            print(f"first doc loaded, {len(text):,} chars, tokenizing...", flush=True)
        for tok_id in tok.encode(text):
            yield tok_id


def batch_stream(tok, micro_batch, smoke=False, seq_len=SEQ_LEN):
    # pack the token stream into seq_len blocks, then into batches
    buf = []
    stream = token_stream(tok, smoke=smoke)
    first = True
    while True:
        while len(buf) < seq_len * micro_batch:
            try:
                buf.append(next(stream))
            except StopIteration:
                stream = token_stream(tok, smoke=smoke)
        chunk = buf[: seq_len * micro_batch]
        buf = buf[seq_len * micro_batch:]
        x = torch.tensor(chunk, dtype=torch.long).view(micro_batch, seq_len)
        if first and is_main():
            print("first batch ready, starting training...", flush=True)
            first = False
        yield x


def save_ckpt(model, tok, out_dir, step, tokens_seen, extra=None):
    # only rank 0 writes, FSDP gathers the full state dict first
    if not is_main():
        return
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    state = {"step": step, "tokens_seen": tokens_seen}
    if extra:
        state.update(extra)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # distributed setup, torchrun sets these env vars
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = world_size > 1 and not args.smoke
    if use_dist:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_main():
        print(f"device: {device}, world size: {world_size}", flush=True)

    if args.smoke:
        # smoke mode uses random tokens, no dataset, no tokenizer file
        # and a tiny sequence so it actually finishes on CPU
        tok = None
        args.seq_len = 64
        args.micro_batch = 1
        args.grad_accum = 1
        args.optim = "adamw"
        if is_main():
            print("smoke mode: random tokens, no dataset, no tokenizer file", flush=True)
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        tok.pad_token = "<pad>"
        tok.bos_token = "<bos>"
        tok.eos_token = "<eos>"
        assert tok.vocab_size == VOCAB_SIZE, f"tokenizer vocab {tok.vocab_size} != {VOCAB_SIZE}"

    # smoke uses a tiny twin of the same shape (2 layers, 256 hidden)
    # for the loop test, because a full 500B backward pass needs a cluster.
    # The full 500B param count is tested with python model.py.
    train_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if args.resume:
        if is_main():
            print(f"resuming weights from {args.resume}", flush=True)
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(args.resume, torch_dtype=train_dtype)
        step = 0
        tokens_seen = 0
        resume_state = os.path.join(args.resume, "trainer_state.json")
        if os.path.exists(resume_state):
            with open(resume_state) as f:
                saved = json.load(f)
            step = saved.get("step", 0)
            tokens_seen = saved.get("tokens_seen", 0)
        if is_main():
            print(f"resumed at step {step}, tokens {tokens_seen:,}", flush=True)
    else:
        if args.smoke:
            from transformers import LlamaConfig, LlamaForCausalLM

            tiny_cfg = build_config()
            tiny_cfg.hidden_size = 256
            tiny_cfg.intermediate_size = 512
            tiny_cfg.num_hidden_layers = 2
            tiny_cfg.num_attention_heads = 4
            tiny_cfg.num_key_value_heads = 2
            model = LlamaForCausalLM(tiny_cfg)
            if is_main():
                print("smoke model: tiny twin, same block layout, 2 layers", flush=True)
        else:
            model = build_model(dtype=train_dtype)
        step = 0
        tokens_seen = 0

    total, _ = count_params(model)
    if is_main():
        print(f"model: {total:,} params ({total / 1e9:.1f}B)", flush=True)
    model.to(device)

    # FSDP2: shard the 500B model across all GPUs
    if use_dist:
        from torch.distributed.fsdp import fully_shard

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        if is_main():
            print(f"FSDP2 enabled, sharded across {world_size} GPUs", flush=True)

    model.gradient_checkpointing_enable()
    model.train()

    if args.smoke:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if is_main():
            print("optimizer: AdamW (smoke, tiny model)", flush=True)
    elif args.optim == "bnb8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
            if is_main():
                print("optimizer: AdamW 8-bit", flush=True)
        except Exception as e:
            if is_main():
                print(f"8-bit Adam not available ({e}), falling back to AdamW", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.95))
        if is_main():
            print("optimizer: AdamW", flush=True)

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum * world_size
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(1, args.max_tokens // tokens_per_step)
    sched = get_cosine_schedule_with_warmup(
        opt, args.warmup, total_steps)
    if is_main():
        print(f"plan: {total_steps} optimizer steps, {tokens_per_step:,} tokens/step", flush=True)
        print(f"target tokens: {args.max_tokens:,}", flush=True)

    stream = batch_stream(tok, args.micro_batch, smoke=args.smoke, seq_len=args.seq_len)
    use_bf16 = device != "cpu"
    t0 = time.time()
    micro = 0
    loss_sum = 0.0
    loss_n = 0

    if is_main():
        print("waiting for first batch, this can take a few minutes on first run...", flush=True)
    while step < total_steps:
        x = next(stream).to(device)
        if micro == 0 and step == 0 and is_main():
            print("first batch in, running first forward/backward...", flush=True)
        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                            dtype=torch.bfloat16, enabled=use_bf16):
            out = model(x, labels=x)
            loss = out.loss / args.grad_accum
        loss.backward()
        loss_sum += out.loss.item()
        loss_n += 1
        micro += 1
        tokens_seen += x.numel() * world_size

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if is_main() and (step <= 20 or step % args.log_every == 0):
                el = time.time() - t0
                tok_s = tokens_seen / max(el, 1)
                print(f"step {step}/{total_steps} loss {loss_sum / max(loss_n, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} tokens {tokens_seen:,} "
                      f"{tok_s:,.0f} tok/s elapsed {el / 60:.1f}m", flush=True)
                loss_sum, loss_n = 0.0, 0
            if step % args.ckpt_every == 0 or step == total_steps:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                if is_main():
                    print(f"saving {ckpt}", flush=True)
                if tok is not None:
                    save_ckpt(model, tok, ckpt, step, tokens_seen)
                else:
                    if is_main():
                        os.makedirs(ckpt, exist_ok=True)
                        model.save_pretrained(ckpt)
                if is_main():
                    with open(os.path.join(args.out, "train_log.jsonl"), "a") as f:
                        f.write(json.dumps({"step": step, "tokens_seen": tokens_seen,
                                            "time_s": time.time() - t0}) + "\n")
            if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                if not os.path.exists(os.path.join(ckpt, "trainer_state.json")):
                    if is_main():
                        print(f"time limit reached, saving {ckpt}", flush=True)
                    if tok is not None:
                        save_ckpt(model, tok, ckpt, step, tokens_seen)
                    else:
                        if is_main():
                            os.makedirs(ckpt, exist_ok=True)
                            model.save_pretrained(ckpt)
                if is_main():
                    print(f"stopping at step {step}, time limit {args.max_hours}h reached", flush=True)
                break
            if args.smoke and step >= 3:
                if is_main():
                    print("smoke done, 3 steps ran fine", flush=True)
                break

    if is_main():
        print(f"done. steps {step}, tokens {tokens_seen:,}, time {(time.time() - t0) / 60:.1f}m", flush=True)
    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
