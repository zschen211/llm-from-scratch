from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_DATEFMT = "%H:%M:%S"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)


def _resolve_src_log_path(cli_arg: str | Path) -> Path:
    """
    沙箱容器内由入口脚本设置 SANDBOX_SRC_LOG_PATH，使任意 CLI 将 llm_from_scratch.* 写入
    runs/<id>/logs/src.log，而不是默认的与 --out 同目录的 *.src.log。
    """
    env_path = os.environ.get("SANDBOX_SRC_LOG_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    p = Path(cli_arg)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def configure_src_stdout_logging(*, level: int = logging.INFO) -> None:
    """
    When running a `src/` python file directly, configure `llm_from_scratch.*`
    logs to stdout using the project-wide unified format.
    """
    pkg_logger = logging.getLogger("llm_from_scratch")
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in pkg_logger.handlers):
        h = logging.StreamHandler(stream=sys.stdout)
        h.setLevel(level)
        h.setFormatter(_formatter())
        pkg_logger.addHandler(h)


def configure_sandbox_entrypoint_logging(*, src_log_path: str | Path, level: int = logging.INFO) -> None:
    """
    容器入口：llm_from_scratch.* 同时写入 logs/src.log 与 stdout（便于 docker logs 与落盘一致）。
    """
    src_log_path = Path(src_log_path)
    src_log_path.parent.mkdir(parents=True, exist_ok=True)

    pkg_logger = logging.getLogger("llm_from_scratch")
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False

    resolved = src_log_path.resolve()
    if not any(
        isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")) == resolved
        for h in pkg_logger.handlers
    ):
        fh = RotatingFileHandler(
            filename=str(resolved),
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(_formatter())
        pkg_logger.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in pkg_logger.handlers
    ):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(_formatter())
        pkg_logger.addHandler(sh)


def configure_cli_stdout_and_src_file_logging(
    *,
    src_log_path: str | Path,
    cli_logger_name: str,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """
    For CLI entrypoints:
    - CLI logger -> stdout
    - `llm_from_scratch.*` (src package) -> rotating file
    Returns the configured CLI logger.
    """
    src_log_path = _resolve_src_log_path(src_log_path)

    # --- src package logger to file ---
    pkg_logger = logging.getLogger("llm_from_scratch")
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False

    # Avoid duplicating handlers across multiple CLI invocations in-process.
    if not any(isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")) == src_log_path for h in pkg_logger.handlers):
        fh = RotatingFileHandler(
            filename=str(src_log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(_formatter())
        pkg_logger.addHandler(fh)

    # --- CLI logger to stdout ---
    cli_logger = logging.getLogger(cli_logger_name)
    cli_logger.setLevel(level)
    cli_logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in cli_logger.handlers):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(_formatter())
        cli_logger.addHandler(sh)

    return cli_logger

