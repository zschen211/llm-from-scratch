from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_DATEFMT = "%H:%M:%S"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)


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
    src_log_path = Path(src_log_path)
    src_log_path.parent.mkdir(parents=True, exist_ok=True)

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

