#!/usr/bin/env python3
"""
CLI：对应 llm_from_scratch.bpe_tokenizer.tokenizer.BPETokenizer 的 public 方法。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def cmd_pretokenize(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizer

    tok = BPETokenizer(
        vocab_version=args.vocab_version,
        force_rebuild=args.force_rebuild,
    )
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    else:
        print("error: 需要 --text 或 --file", file=sys.stderr)
        return 1
    out = tok.pretokenize(text)
    if args.format == "json":
        print(json.dumps(out))
    else:
        print(out)
    return 0


def cmd_vocab_info(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import BPETokenizer

    tok = BPETokenizer(
        vocab_version=args.vocab_version,
        force_rebuild=args.force_rebuild,
    )
    print(len(tok.vocab))
    print(tok.WORD_END_TOKEN_ID)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BPETokenizer CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pretokenize", help="pretokenize(text_chunk)")
    sp.add_argument("--text", type=str, default=None)
    sp.add_argument("--file", type=Path, default=None)
    sp.add_argument("--vocab-version", type=int, default=1)
    sp.add_argument("--force-rebuild", action="store_true")
    sp.add_argument(
        "--format",
        choices=("repr", "json"),
        default="repr",
        help="输出格式",
    )
    sp.set_defaults(func=cmd_pretokenize)

    sp = sub.add_parser("vocab-info", help="打印 vocab 长度与 WORD_END_TOKEN_ID")
    sp.add_argument("--vocab-version", type=int, default=1)
    sp.add_argument("--force-rebuild", action="store_true")
    sp.set_defaults(func=cmd_vocab_info)

    return p


def main(argv: list[str] | None = None) -> int:
    _project_root()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
