from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from llm_from_scratch.sandbox.sandbox_runner_metrics import PrometheusMetricsCollector


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    logs_dir = out_dir / "logs"
    metrics_dir = out_dir / "metrics"
    prof_dir = out_dir / "prof"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, metrics_dir, prof_dir


def _run_command_and_log(cmd: list[str], log_path: Path, cwd: Path) -> int:
    # 合并 stdout/stderr，写到同一个日志里（也会回显到容器 stdout，便于 docker logs 查看）。
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Container entrypoint for sandbox runner.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--metrics-interval-s", type=float, default=1.0)
    parser.add_argument("--", dest="cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if not args.cmd:
        print("Error: no command provided after '--'.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    logs_dir, _, _ = _ensure_dirs(out_dir)

    run_id = os.environ.get("SANDBOX_RUN_ID", out_dir.name)
    cpu_limit_env = os.environ.get("SANDBOX_CPU_LIMIT")
    cpu_limit = float(cpu_limit_env) if cpu_limit_env else None

    metrics_collector = PrometheusMetricsCollector(
        run_id=run_id,
        out_metrics_path=args.metrics_path,
        interval_s=args.metrics_interval_s,
        metrics_http_port=args.metrics_port,
        cpu_limit=cpu_limit,
    )

    metrics_collector.start()

    # 默认在 /workspace 下执行，这也是项目源码/CLI 常见相对路径基准。
    cwd = Path("/workspace")
    log_path = logs_dir / "command.log"

    try:
        exit_code = _run_command_and_log(args.cmd, log_path=log_path, cwd=cwd)
    finally:
        metrics_collector.stop()

    # 把 exit code 记录下来，便于后处理。
    (out_dir / "exit_code.txt").write_text(str(exit_code), encoding="utf-8")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

