"""Train the AIMYTOG BPE tokenizer. 152064 tokens, byte-level BPE.

Usage:
  python train_tokenizer.py --data ./data/train.txt --out ./tokenizer
  python train_tokenizer.py --data ./data/*.txt --vocab-size 152064

The tokenizer is trained on your own text. Give it a large, clean
corpus and it will learn the merges. A good tokenizer makes training
faster and the model smarter, because fewer tokens means more text
per context window.
"""
import argparse
import glob
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 152064
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True,
                   help="text files to train on, globs allowed")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = p.parse_args()

    files = []
    for pat in args.data:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit("no input files found")
    print(f"training on {len(files)} files", flush=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL,
        min_frequency=2,
        show_progress=True,
    )
    tok.train(files, trainer)

    os.makedirs(args.out, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast.save_pretrained(args.out)
    print(f"saved tokenizer to {args.out}", flush=True)
    print(f"vocab size: {len(fast)}", flush=True)


if __name__ == "__main__":
    main()
