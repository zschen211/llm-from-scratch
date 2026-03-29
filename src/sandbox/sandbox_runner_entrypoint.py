from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from llm_from_scratch._logging import configure_sandbox_entrypoint_logging

_log = logging.getLogger(__name__)


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    logs_dir = out_dir / "logs"
    prof_dir = out_dir / "prof"
    logs_dir.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, prof_dir


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
    # 用户命令：用标准的 REMAINDER 接收（由调用方通过 `--` 与本入口参数分隔）
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    logs_dir, _ = _ensure_dirs(out_dir)

    # docker run 会注入 SANDBOX_SRC_LOG_PATH；本地直接跑入口脚本时在此补全。
    if "SANDBOX_SRC_LOG_PATH" not in os.environ:
        os.environ["SANDBOX_SRC_LOG_PATH"] = str((logs_dir / "src.log").resolve())
    src_log_path = Path(os.environ["SANDBOX_SRC_LOG_PATH"]).resolve()
    src_log_path.parent.mkdir(parents=True, exist_ok=True)
    src_log_path.touch(exist_ok=True)
    configure_sandbox_entrypoint_logging(src_log_path=src_log_path, level=logging.INFO)

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        _log.error("no command provided")
        return 2

    # 容器内默认在 /workspace 下执行；如果直接在宿主机跑本脚本（调试），
    # 则回退到当前工作目录，避免 /workspace 不存在导致失败。
    cwd = Path("/workspace")
    if not cwd.is_dir():
        cwd = Path.cwd()
    log_path = logs_dir / "command.log"

    _log.info("running command: %s", " ".join(cmd))
    exit_code = _run_command_and_log(cmd, log_path=log_path, cwd=cwd)

    # 把 exit code 记录下来，便于后处理。
    (out_dir / "exit_code.txt").write_text(str(exit_code), encoding="utf-8")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
