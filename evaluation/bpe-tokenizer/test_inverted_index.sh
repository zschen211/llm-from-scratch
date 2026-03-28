#!/usr/bin/env bash
# 测试倒排索引优化效果

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_FILE="$PROJECT_ROOT/data/TinyStoriesV2-GPT4-train-50M.txt"
VOCAB_SIZE=5000
SPECIAL_TOKEN="<|endoftext|>"

echo "=== Testing BPE Training with Inverted Index Optimization ==="
echo "Input: $INPUT_FILE"
echo "Vocab size: $VOCAB_SIZE"
echo ""

# Test 1: With inverted index (default)
echo "--- Test 1: WITH inverted index ---"
time python "$PROJECT_ROOT/cli/bpe_tokenizer/train_bpe_cli.py" \
  --input-corpus "$INPUT_FILE" \
  --vocab-size "$VOCAB_SIZE" \
  --special-token "$SPECIAL_TOKEN" \
  --out "$PROJECT_ROOT/.tmp/tokenizer_with_index.json" \
  --stream-chunk-chars 1000000 \
  --stream-memory-target-percent 70 \
  --min-pair-freq 2 \
  --use-inverted-index \
  --no-print-metrics

echo ""
echo "--- Test 2: WITHOUT inverted index (baseline) ---"
time python "$PROJECT_ROOT/cli/bpe_tokenizer/train_bpe_cli.py" \
  --input-corpus "$INPUT_FILE" \
  --vocab-size "$VOCAB_SIZE" \
  --special-token "$SPECIAL_TOKEN" \
  --out "$PROJECT_ROOT/.tmp/tokenizer_without_index.json" \
  --stream-chunk-chars 1000000 \
  --stream-memory-target-percent 70 \
  --min-pair-freq 2 \
  --no-inverted-index \
  --no-print-metrics

echo ""
echo "=== Comparison Complete ==="
echo "Check the timing results above to see the speedup from inverted index."
