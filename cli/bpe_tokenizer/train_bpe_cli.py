from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 运行 `python cli/.../xxx_cli.py` 时，`sys.path[0]` 指向脚本目录，
# 需要显式把项目根目录加入 sys.path，才能导入真实包。
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.bpe_tokenizer import train_bpe
from llm_from_scratch._logging import (
    DEFAULT_DATEFMT,
    DEFAULT_FORMAT,
    configure_cli_stdout_and_src_file_logging,
)


def _save_checkpoint(path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    payload = {
        "vocab": {str(k): list(v) for k, v in vocab.items()},
        "merges": [[list(a), list(b)] for a, b in merges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="train_bpe CLI: train vocab/merges and write checkpoint JSON")
    parser.add_argument("--input-corpus", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Total vocab size including special tokens.")
    parser.add_argument(
        "--special-token",
        type=str,
        action="append",
        default=[],
        help="Special token string (can be repeated).",
    )
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint JSON path.")

    parser.add_argument(
        "--disable-packaged-regression",
        action="store_true",
        help="Disable packaged regression for corpus.en (for debug).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker count for pair counting/merge (set 1 for serial).",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="(无效) Rust-only 训练无 checkpoint 恢复；保留仅为兼容旧脚本。",
    )
    parser.add_argument(
        "--no-print-metrics",
        action="store_true",
        help="兼容旧脚本。进度见 src.log / llm_from_scratch 日志（bpe_core pretokenize/merge）。",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="(无效) Rust-only 训练不支持中途 checkpoint；保留仅为兼容旧脚本。",
    )
    parser.add_argument(
        "--checkpoint-every-n-merges",
        type=int,
        default=50,
        help="(无效) Rust-only 训练不支持；保留仅为兼容旧脚本。",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling; saves .prof file to --profile-dir (default: .prof/).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory for .prof output (implies --profile). Default: <project_root>/.prof/",
    )
    parser.add_argument(
        "--stream-chunk-chars",
        type=int,
        default=1_000_000,
        help="Streaming: Rust 每次从文件读取的字节上限（默认 1000000，参数名沿用历史）。必须 >0。",
    )
    parser.add_argument(
        "--stream-memory-target-percent",
        type=float,
        default=85.0,
        help="(无效) Rust 内部策略；忽略。",
    )
    parser.add_argument(
        "--min-pair-freq",
        type=int,
        default=1,
        help="(无效) Rust 使用固定策略；非 1 时 train_bpe 会记警告。",
    )
    parser.add_argument(
        "--use-inverted-index",
        action="store_true",
        default=True,
        help="(无效) Rust 内部实现；忽略。",
    )
    parser.add_argument(
        "--no-inverted-index",
        action="store_false",
        dest="use_inverted_index",
        help="(无效) Rust-only；忽略。",
    )

    args = parser.parse_args(argv)

    src_log_path = Path(args.out).with_suffix(".src.log")
    cli_log = configure_cli_stdout_and_src_file_logging(
        src_log_path=src_log_path,
        cli_logger_name="cli.bpe_tokenizer.train_bpe",
        level=logging.INFO,
    )
    cli_log.info("train_bpe_cli start: input=%s vocab_size=%d out=%s", args.input_corpus, args.vocab_size, args.out)
    cli_log.info("src logs -> %s", src_log_path)
    # Rust `pyo3-log` 使用 target `llm_from_scratch.bpe_core.*`；在包 logger 上挂 stdout，子 logger 向上传播后终端与 src.log 一致。
    pkg_logger = logging.getLogger("llm_from_scratch")
    if not args.no_print_metrics and not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in pkg_logger.handlers
    ):
        _pkg_sh = logging.StreamHandler(stream=sys.stdout)
        _pkg_sh.setLevel(logging.INFO)
        _pkg_sh.setFormatter(logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
        pkg_logger.addHandler(_pkg_sh)
    if not args.no_print_metrics:
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
        cli_log.info(
            "Rust 训练进度写入本终端与 %s（logger: llm_from_scratch.*，含 bpe_core）",
            src_log_path,
        )

    profile_dir = None
    if args.profile or args.profile_dir:
        profile_dir = args.profile_dir or str(_project_root / ".prof")

    if args.checkpoint_path:
        cli_log.warning("--checkpoint-path 在 Rust-only 训练下无效，已忽略")
    if args.force_restart:
        cli_log.warning("--force-restart 在 Rust-only 训练下无效，已忽略")

    if args.stream_chunk_chars <= 0:
        cli_log.error("--stream-chunk-chars 必须 > 0（Rust 无全量读入模式）")
        return 2

    vocab, merges = train_bpe(
        input_path=Path(args.input_corpus),
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        disable_packaged_regression=bool(args.disable_packaged_regression),
        num_workers=args.num_workers,
        profile_dir=profile_dir,
        stream_chunk_chars=args.stream_chunk_chars,
        min_pair_freq=args.min_pair_freq,
        stream_memory_target_percent=args.stream_memory_target_percent,
        use_inverted_index=args.use_inverted_index,
    )

    _save_checkpoint(Path(args.out), vocab=vocab, merges=merges)
    cli_log.info("train_bpe_cli done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

