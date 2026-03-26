"""
sandbox_runner_cli：在 docker 沙盒内执行项目子命令并采集监控数据。

主要能力：
- 以用户给定的 CPU/内存限制运行 docker 容器
- 把宿主机 `data/` 以只读方式挂载到容器（`/workspace/data`）
- 容器内启动一个简易 Prometheus 指标采集器：
  - 指标 HTTP：`http://<host>:<metrics-host-port>/metrics`
  - 指标落盘：`--output-root/<run-id>/metrics/metrics.prom`
- 记录命令日志：`--output-root/<run-id>/logs/command.log`
- profiling 落盘：如果子命令支持 `--prof/--profile` 等选项，输出会写入挂载目录
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

# 运行 `python cli/.../xxx_cli.py` 时，`sys.path[0]` 指向脚本目录，
# 需要显式把项目根目录加入 sys.path，才能导入真实包。
#
# 本文件路径为：<project_root>/cli/sandbox/sandbox_runner_cli.py
# 因此 project_root = parents[2]。
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.sandbox.sandbox_runner import SandboxRunConfig, run_sandbox  # noqa: E402
from llm_from_scratch._logging import configure_cli_stdout_and_src_file_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sandbox runner: build a docker image and run a command in an isolated resource-limited container."
    )
    parser.add_argument("--cpu", type=float, default=1.0, help="CPU limit for docker (--cpus).")
    parser.add_argument("--memory", type=str, default="512m", help="Memory limit for docker (--memory).")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id used for isolating output directory. Default: random uuid4.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for sandbox outputs (metrics/logs/prof). Default: <project_root>/runs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Host data directory to mount into container as /workspace/data. Default: <project_root>/data",
    )
    parser.add_argument(
        "--dockerfile",
        type=str,
        default=None,
        help="Dockerfile path. Default: <project_root>/docker/Dockerfile.sandbox",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        default="llm-from-scratch-sandbox:dev",
        help="Docker image tag used for sandbox runner.",
    )
    parser.add_argument(
        "--metrics-host-port",
        type=int,
        default=19090,
        help="Expose container metrics HTTP port to this host port.",
    )
    parser.add_argument(
        "--metrics-interval-s",
        type=float,
        default=1.0,
        help="Metrics sampling interval in seconds (container side).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker build step (assumes image already exists).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call docker; just create run directory and write docker commands.",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run inside the sandbox container. Use `--` to separate from options.",
    )

    args = parser.parse_args(argv)

    if not args.cmd:
        parser.error("missing command: provide a command after `--`")

    # argparse.REMAINDER 会把分隔符 `--` 也包含进列表；这里去掉，避免容器侧再出现 `-- -- cmd`。
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    run_id = args.run_id or str(uuid.uuid4())
    output_root = (Path(args.output_root) if args.output_root else (_project_root / "runs")).resolve()
    data_dir = (Path(args.data_dir) if args.data_dir else (_project_root / "data")).resolve()
    dockerfile = (Path(args.dockerfile) if args.dockerfile else (_project_root / "docker" / "Dockerfile.sandbox")).resolve()

    src_log_path = (output_root / run_id / "logs" / "src.log").resolve()
    cli_log = configure_cli_stdout_and_src_file_logging(
        src_log_path=src_log_path,
        cli_logger_name="cli.sandbox.sandbox_runner",
        level=logging.INFO,
    )

    cfg = SandboxRunConfig(
        cmd=cmd,
        cpu=args.cpu,
        memory=args.memory,
        run_id=run_id,
        output_root=output_root,
        project_root=_project_root,
        dockerfile=dockerfile,
        image_tag=args.image_tag,
        data_dir=data_dir,
        metrics_host_port=args.metrics_host_port,
        dry_run=bool(args.dry_run),
        build_image=not bool(args.skip_build),
        metrics_interval_s=args.metrics_interval_s,
    )

    def _log(msg: str) -> None:
        # run_sandbox expects a callable; we keep it line-based for streaming output.
        cli_log.info("%s", msg.rstrip("\n"))

    cli_log.info("sandbox_runner_cli start: run_id=%s cpu=%s memory=%s", run_id, cfg.cpu, cfg.memory)
    cli_log.info("cmd: %s", " ".join(cmd))
    cli_log.info("output: %s", output_root / run_id)
    cli_log.info("metrics: http://localhost:%d/metrics", args.metrics_host_port)
    cli_log.info("src logs -> %s", src_log_path)

    result = run_sandbox(cfg, log=_log)

    cli_log.info("exit_code: %s", result.exit_code)
    cli_log.info("logs: %s", result.logs_dir / "command.log")
    cli_log.info("metrics: %s", result.metrics_dir / "metrics.prom")

    return int(result.exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

