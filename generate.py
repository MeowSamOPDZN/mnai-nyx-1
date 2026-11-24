"""Generate text with a trained AIMYTOG checkpoint.

Usage:
  python generate.py --model out-aimytog/ckpt-1000 --prompt "The future of AI is"
  python generate.py --model out-aimytog/ckpt-1000 --prompt "Once upon a time" --max-new 200 --temp 0.8

A 500B checkpoint is 1TB in bf16. It does not fit on one GPU.
Use tensor parallelism, pipeline parallelism, or a sharded inference
engine like vLLM or TensorRT-LLM to serve it. This script is for a
single-node test with a small checkpoint, or for loading a sharded
checkpoint that already fits in your GPU memory.
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--max-new", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    print(f"loaded {args.model}", flush=True)

    ids = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new,
            do_sample=True,
            temperature=args.temp,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
