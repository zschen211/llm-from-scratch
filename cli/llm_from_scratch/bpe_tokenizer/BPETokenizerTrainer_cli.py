#!/usr/bin/env python3
"""
CLI：对应 llm_from_scratch.bpe_tokenizer.tokenizer.BPETokenizerTrainer 的 public 方法。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def cmd_preview(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizerTrainer

    with BPETokenizerTrainer(
        args.file,
        vocab_version=args.vocab_version,
        force_rebuild=args.force_rebuild,
        n_processes=args.n_processes,
    ) as trainer:
        chunk = trainer.read_chunk(args.chunk_size)
        frags = trainer.tokenizer.pretokenize(chunk)
        print(len(chunk))
        print(len(frags))
        if args.head:
            print(frags[: args.head])
    return 0


def cmd_cursor(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizerTrainer

    with BPETokenizerTrainer(
        args.file,
        vocab_version=args.vocab_version,
        n_processes=1,
    ) as trainer:
        print(trainer.get_cursor())
        trainer.read_chunk(args.read_size)
        print(trainer.get_cursor())
        trainer.reset_cursor()
        print(trainer.get_cursor())
    return 0


def cmd_parallel(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizerTrainer
    from cli.llm_from_scratch.bpe_tokenizer.trainer_pool_worker import add_one

    with BPETokenizerTrainer(
        args.file,
        vocab_version=args.vocab_version,
        force_rebuild=args.force_rebuild,
        n_processes=args.n_processes,
    ) as trainer:
        out = list(trainer.map_parallel(add_one, args.inputs, chunksize=args.chunksize))
    print(out)
    return 0


def cmd_len(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizerTrainer

    with BPETokenizerTrainer(args.file, n_processes=1) as trainer:
        print(len(trainer))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BPETokenizerTrainer CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("preview", help="read_chunk + tokenizer.pretokenize，打印统计与前若干片段")
    sp.add_argument("--file", required=True, type=Path)
    sp.add_argument("--chunk-size", type=int, default=8192)
    sp.add_argument("--vocab-version", type=int, default=1)
    sp.add_argument("--force-rebuild", action="store_true")
    sp.add_argument("--n-processes", type=int, default=1)
    sp.add_argument("--head", type=int, default=0, help=">0 时打印前 N 个片段")
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser(
        "cursor-flow",
        help="演示 get_cursor / read_chunk / reset_cursor 序列",
    )
    sp.add_argument("--file", required=True, type=Path)
    sp.add_argument("--read-size", type=int, default=3)
    sp.add_argument("--vocab-version", type=int, default=1)
    sp.set_defaults(func=cmd_cursor)

    sp = sub.add_parser("parallel", help="map_parallel(add_one, inputs)")
    sp.add_argument("--file", required=True, type=Path, help="Trainer 所需存在的训练文件（可为占位）")
    sp.add_argument(
        "--inputs",
        type=int,
        nargs="+",
        required=True,
        help="整数列表，经 add_one 映射",
    )
    sp.add_argument("--n-processes", type=int, default=2)
    sp.add_argument("--vocab-version", type=int, default=1)
    sp.add_argument("--force-rebuild", action="store_true")
    sp.add_argument("--chunksize", type=int, default=1)
    sp.set_defaults(func=cmd_parallel)

    sp = sub.add_parser("len", help="打印 len(trainer)（文件字节数）")
    sp.add_argument("--file", required=True, type=Path)
    sp.set_defaults(func=cmd_len)

    return p


def main(argv: list[str] | None = None) -> int:
    _project_root()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
