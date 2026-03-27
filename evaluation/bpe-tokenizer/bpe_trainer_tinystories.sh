#!/usr/bin/env sh

# 资源配置（可通过环境变量覆盖）
CPU_LIMIT=${CPU_LIMIT:-6}
MEMORY_LIMIT=${MEMORY_LIMIT:-10g}
METRICS_PORT=${METRICS_PORT:-19090}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}

# 训练参数
VOCAB_SIZE=${VOCAB_SIZE:-10000}
STREAM_MEMORY_PERCENT=${STREAM_MEMORY_PERCENT:-85}
# 对于大文件（2GB+），建议设置 MIN_PAIR_FREQ=2 以减少内存占用
MIN_PAIR_FREQ=${MIN_PAIR_FREQ:-2}
ENABLE_PROFILE=${ENABLE_PROFILE:-true}

echo "=== BPE Tokenizer Training Configuration ==="
echo "CPU Limit: $CPU_LIMIT cores"
echo "Memory Limit: $MEMORY_LIMIT"
echo "Vocab Size: $VOCAB_SIZE"
echo "Stream Memory Target: $STREAM_MEMORY_PERCENT%"
echo "Min Pair Frequency: $MIN_PAIR_FREQ (recommended 2+ for large files)"
echo "Profile Enabled: $ENABLE_PROFILE"
echo "==========================================="

PROFILE_FLAG=""
if [ "$ENABLE_PROFILE" = "true" ]; then
  PROFILE_FLAG="--profile"
fi

uv run python cli/sandbox/sandbox_runner_cli.py \
  --cpu "$CPU_LIMIT" \
  --memory "$MEMORY_LIMIT" \
  --metrics-host-port "$METRICS_PORT" \
  --output-root "$OUTPUT_ROOT" \
  -- python cli/bpe_tokenizer/train_bpe_cli.py \
    --input-corpus data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size "$VOCAB_SIZE" \
    --special-token "<|endoftext|>" \
    --out tokenizer/bpe_tokenizer_tinystories_train.json \
    --stream-memory-target-percent "$STREAM_MEMORY_PERCENT" \
    --min-pair-freq "$MIN_PAIR_FREQ" \
    $PROFILE_FLAG
