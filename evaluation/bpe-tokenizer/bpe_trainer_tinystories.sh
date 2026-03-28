#!/usr/bin/env sh

# 资源配置（可通过环境变量覆盖）
CPU_LIMIT=${CPU_LIMIT:-6}
MEMORY_LIMIT=${MEMORY_LIMIT:-10g}
METRICS_PORT=${METRICS_PORT:-19090}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-bpe-train-$(date +%Y%m%d-%H%M%S)}

# 训练参数
VOCAB_SIZE=${VOCAB_SIZE:-270}
STREAM_MEMORY_PERCENT=${STREAM_MEMORY_PERCENT:-85}
# 对于大文件（2GB+），建议设置 MIN_PAIR_FREQ=2 以减少内存占用
MIN_PAIR_FREQ=${MIN_PAIR_FREQ:-2}
ENABLE_PROFILE=${ENABLE_PROFILE:-true}
# perf trace 可选 syscall 过滤，例如：openat,read,write,close,futex
PERF_TRACE_FILTER=${PERF_TRACE_FILTER:-}
PERF_TRACE_SUMMARY=${PERF_TRACE_SUMMARY:-true}
# Ctrl+C 后给子进程一点时间优雅退出，确保 perf trace 尽量落盘完整。
PERF_FLUSH_GRACE_S=${PERF_FLUSH_GRACE_S:-3}

CHILD_PID=""

_graceful_stop_child() {
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "[signal] forwarding SIGINT to child pid=$CHILD_PID ..."
    kill -INT "$CHILD_PID" 2>/dev/null || true
    sleep "$PERF_FLUSH_GRACE_S"
    if kill -0 "$CHILD_PID" 2>/dev/null; then
      echo "[signal] child still alive, forwarding SIGTERM ..."
      kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
  fi
}

_on_interrupt() {
  echo "[signal] interrupt received, waiting for graceful shutdown ..."
  _graceful_stop_child
  if [ -n "$CHILD_PID" ]; then
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit 130
}

trap '_on_interrupt' INT TERM

echo "=== BPE Tokenizer Training Configuration ==="
echo "Run ID: $RUN_ID"
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
  --run-id "$RUN_ID" \
  -- python cli/bpe_tokenizer/train_bpe_cli.py \
    --input-corpus data/TinyStoriesV2-GPT4-train-50M.txt \
    --vocab-size "$VOCAB_SIZE" \
    --special-token "<|endoftext|>" \
    --out tokenizer/bpe_tokenizer_tinystories_train.json \
    --stream-memory-target-percent "$STREAM_MEMORY_PERCENT" \
    --min-pair-freq "$MIN_PAIR_FREQ" \
    $PROFILE_FLAG &
CHILD_PID=$!

wait "$CHILD_PID"
EXIT_CODE=$?
trap - INT TERM
exit "$EXIT_CODE"
