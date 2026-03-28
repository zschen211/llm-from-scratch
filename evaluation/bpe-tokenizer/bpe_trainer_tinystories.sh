#!/usr/bin/env sh

# 资源配置（可通过环境变量覆盖）
CPU_LIMIT=${CPU_LIMIT:-6}
MEMORY_LIMIT=${MEMORY_LIMIT:-10g}
METRICS_PORT=${METRICS_PORT:-19090}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-bpe-train-$(date +%Y%m%d-%H%M%S)}

# 训练参数
VOCAB_SIZE=${VOCAB_SIZE:-10000}
STREAM_MEMORY_PERCENT=${STREAM_MEMORY_PERCENT:-85}
# 对于大文件（2GB+），建议设置 MIN_PAIR_FREQ=2 以减少内存占用
MIN_PAIR_FREQ=${MIN_PAIR_FREQ:-2}
ENABLE_PROFILE=${ENABLE_PROFILE:-true}
ENABLE_PERF_TRACE=${ENABLE_PERF_TRACE:-true}
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
echo "Perf Trace Enabled: $ENABLE_PERF_TRACE"
echo "==========================================="

PROFILE_FLAG=""
if [ "$ENABLE_PROFILE" = "true" ]; then
  PROFILE_FLAG="--profile"
fi

# 可选：在容器内对 train_bpe_cli 做 perf trace（较 strace 开销更低）。
# 说明：
# - 输出文件位于宿主机 runs/<run_id>/logs/perf/perf-trace.log（容器内为 $SANDBOX_OUT_DIR/logs/perf/）
# - 若使用 /workspace/runs/... 会错：sandbox 只把本次运行目录挂到 /sandbox_out，与仓库内 runs 不是同一路径。
# - 如果容器里没有 perf，会自动降级为普通运行。
# - sandbox_runner 默认已为容器添加 PERFMON，perf trace 可直接使用（也可用 --no-sandbox-perf-caps 关闭）。
if [ "$ENABLE_PERF_TRACE" = "true" ]; then
  PERF_SUMMARY_FLAG=""
  if [ "$PERF_TRACE_SUMMARY" = "true" ]; then
    PERF_SUMMARY_FLAG="-s"
  fi
  uv run python cli/sandbox/sandbox_runner_cli.py \
    --cpu "$CPU_LIMIT" \
    --memory "$MEMORY_LIMIT" \
    --metrics-host-port "$METRICS_PORT" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$RUN_ID" \
    -- sh -lc "PERF_DIR=\"\${SANDBOX_OUT_DIR:-/sandbox_out}/logs/perf\" && mkdir -p \"\$PERF_DIR\" && \
TRAIN_CMD=\"python cli/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data/TinyStoriesV2-GPT4-train-50M.txt \
  --vocab-size $VOCAB_SIZE \
  --special-token '<|endoftext|>' \
  --out tokenizer/bpe_tokenizer_tinystories_train.json \
  --stream-memory-target-percent $STREAM_MEMORY_PERCENT \
  --min-pair-freq $MIN_PAIR_FREQ \
  $PROFILE_FLAG\" && \
if command -v perf >/dev/null 2>&1; then \
  echo \"[perf-trace] enabled, output file: \$PERF_DIR/perf-trace.log\"; \
  if [ -n \"$PERF_TRACE_FILTER\" ]; then \
    perf trace $PERF_SUMMARY_FLAG -e \"$PERF_TRACE_FILTER\" -o \"\$PERF_DIR/perf-trace.log\" -- \
      sh -c \"\$TRAIN_CMD\"; \
  else \
    perf trace $PERF_SUMMARY_FLAG -o \"\$PERF_DIR/perf-trace.log\" -- \
      sh -c \"\$TRAIN_CMD\"; \
  fi; \
  PERF_RC=\$?; \
  if [ \"\$PERF_RC\" -ne 0 ]; then \
    echo \"[perf-trace] failed (exit=\$PERF_RC), fallback to normal run; check \$PERF_DIR/perf-trace.log\"; \
    sh -c \"\$TRAIN_CMD\"; \
  fi; \
else \
  echo '[perf-trace] perf not found in container, running without trace'; \
  sh -c \"\$TRAIN_CMD\"; \
fi" &
  CHILD_PID=$!
else
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
fi

wait "$CHILD_PID"
EXIT_CODE=$?
trap - INT TERM
exit "$EXIT_CODE"
