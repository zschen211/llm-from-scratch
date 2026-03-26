#!/usr/bin/env sh

uv run python cli/sandbox/sandbox_runner_cli.py \
  --cpu 8 \
  --memory 8g \
  --metrics-host-port 19090 \
  --output-root runs \
  -- python cli/bpe_tokenizer/train_bpe_cli.py \
    --input-corpus data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --special-token "<|endoftext|>" \
    --out tokenizer/bpe_tokenizer_tinystories_train.json \
    --stream-memory-target-percent 85 \
    --profile