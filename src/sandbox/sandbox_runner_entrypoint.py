from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from llm_from_scratch.sandbox.sandbox_runner_metrics import PrometheusMetricsCollector
from llm_from_scratch._logging import configure_src_stdout_logging

_log = logging.getLogger(__name__)


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
    # 用户命令：用标准的 REMAINDER 接收（由调用方通过 `--` 与本入口参数分隔）
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    # If someone runs this file directly, we want src logs visible on stdout.
    configure_src_stdout_logging(level=logging.INFO)

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        _log.error("no command provided")
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
    _log.info("metrics collector started: port=%d path=%s interval=%.2fs", args.metrics_port, args.metrics_path, args.metrics_interval_s)

    # 容器内默认在 /workspace 下执行；如果直接在宿主机跑本脚本（调试），
    # 则回退到当前工作目录，避免 /workspace 不存在导致失败。
    cwd = Path("/workspace")
    if not cwd.is_dir():
        cwd = Path.cwd()
    log_path = logs_dir / "command.log"

    try:
        _log.info("running command: %s", " ".join(cmd))
        exit_code = _run_command_and_log(cmd, log_path=log_path, cwd=cwd)
    finally:
        metrics_collector.stop()
        _log.info("metrics collector stopped")

    # 把 exit code 记录下来，便于后处理。
    (out_dir / "exit_code.txt").write_text(str(exit_code), encoding="utf-8")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

