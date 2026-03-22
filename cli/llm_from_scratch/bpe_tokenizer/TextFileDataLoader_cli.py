#!/usr/bin/env python3
"""
CLI：对应 llm_from_scratch.bpe_tokenizer.tokenizer.TextFileDataLoader 的 public 方法。
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


def cmd_info(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import TextFileDataLoader

    loader = TextFileDataLoader(args.file)
    print(len(loader))
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import TextFileDataLoader

    loader = TextFileDataLoader(args.file)
    if args.offset is not None:
        loader.set_cursor(args.offset)
    text = loader.read_chunk(args.size)
    sys.stdout.write(text)
    if args.print_cursor:
        print(file=sys.stderr)
        print(loader.get_cursor(), file=sys.stderr)
    return 0


def cmd_cursor_get(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import TextFileDataLoader

    loader = TextFileDataLoader(args.file)
    if args.offset is not None:
        loader.set_cursor(args.offset)
    print(loader.get_cursor())
    return 0


def cmd_cursor_set(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import TextFileDataLoader

    loader = TextFileDataLoader(args.file)
    loader.set_cursor(args.position)
    print(loader.get_cursor())
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    from llm_from_scratch.bpe_tokenizer.tokenizer import TextFileDataLoader

    loader = TextFileDataLoader(args.file)
    loader.set_cursor(999)  # 任意非零
    loader.reset_cursor()
    print(loader.get_cursor())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TextFileDataLoader CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("info", help="打印文件字节长度 __len__(loader)")
    sp.add_argument("--file", required=True, type=Path)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("read", help="read_chunk(size)，可选先 set_cursor(offset)")
    sp.add_argument("--file", required=True, type=Path)
    sp.add_argument("--size", required=True, type=int)
    sp.add_argument("--offset", type=int, default=None)
    sp.add_argument("--print-cursor", action="store_true", help="读完后将游标打印到 stderr")
    sp.set_defaults(func=cmd_read)

    sp = sub.add_parser("cursor-get", help="get_cursor()，可选先 set_cursor(offset)")
    sp.add_argument("--file", required=True, type=Path)
    sp.add_argument("--offset", type=int, default=None)
    sp.set_defaults(func=cmd_cursor_get)

    sp = sub.add_parser("cursor-set", help="set_cursor(position) 后打印游标")
    sp.add_argument("--file", required=True, type=Path)
    sp.add_argument("--position", required=True, type=int)
    sp.set_defaults(func=cmd_cursor_set)

    sp = sub.add_parser("reset", help="reset_cursor() 后打印游标（应先错位再 reset 以验证）")
    sp.add_argument("--file", required=True, type=Path)
    sp.set_defaults(func=cmd_reset)

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
