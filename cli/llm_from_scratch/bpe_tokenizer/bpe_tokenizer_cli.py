from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 运行 `python cli/.../xxx_cli.py` 时，`sys.path[0]` 指向脚本目录，
# 需要显式把项目根目录加入 sys.path，才能导入真实包。
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.bpe_tokenizer import BPETokenizer, make_tokenizer, train_bpe
from llm_from_scratch.bpe_tokenizer._gpt2_bytes import gpt2_bytes_to_unicode


def _load_checkpoint(path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    vocab = {int(k): bytes(v) for k, v in payload["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in payload["merges"]]
    return vocab, merges


def _load_from_gpt2_vocab_merges(
    vocab_json_path: Path, merges_txt_path: Path
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # GPT-2 "vocab.json" maps token_str -> id where token_str is a unicode rendering of bytes.
    # We convert it back to raw bytes to build our BPETokenizer vocab.
    dec = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_json_path, encoding="utf-8") as f:
        gpt2_vocab: dict[str, int] = json.load(f)

    vocab: dict[int, bytes] = {int(idx): bytes([dec[ch] for ch in tok]) for tok, idx in gpt2_vocab.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            t1, t2 = line.split(" ")
            merges.append((bytes([dec[ch] for ch in t1]), bytes([dec[ch] for ch in t2])))

    return vocab, merges


def _build_tokenizer_from_args(args: argparse.Namespace) -> BPETokenizer:
    special_tokens: list[str] = list(args.special_token or [])

    if args.checkpoint:
        vocab, merges = _load_checkpoint(Path(args.checkpoint))
    elif args.gpt2_vocab and args.gpt2_merges:
        vocab, merges = _load_from_gpt2_vocab_merges(Path(args.gpt2_vocab), Path(args.gpt2_merges))
    elif args.train_corpus:
        # Convenience: allow a quick way to produce vocab/merges for CLI usage.
        vocab, merges = train_bpe(
            input_path=Path(args.train_corpus),
            vocab_size=int(args.vocab_size),
            special_tokens=special_tokens,
        )
    else:
        raise SystemExit("Must provide either --checkpoint, or both --gpt2-vocab and --gpt2-merges, or --train-corpus.")

    return make_tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def _read_chunks_text(path: Path, chunk_size: int | None):
    # Important: when chunk_size is provided, we use f.read(chunk_size) to avoid inserting/removing characters.
    # When chunk_size is not provided, we iterate by lines.
    with open(path, "r", encoding="utf-8") as f:
        if chunk_size is None:
            for line in f:
                yield line
        else:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BPETokenizer CLI: encode / decode / encode_iterable")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # 把 tokenizer 相关参数挂到子命令上，保证支持形如：
    #   python ... bpe_tokenizer_cli.py encode --checkpoint tok.json --special-token ...
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=str, default=None, help="Tokenizer checkpoint JSON (our format).")
    common.add_argument("--gpt2-vocab", type=str, default=None, help="GPT-2 vocab.json (token_str -> id).")
    common.add_argument("--gpt2-merges", type=str, default=None, help="GPT-2 merges.txt (space-separated pairs).")
    common.add_argument("--train-corpus", type=str, default=None, help="If set, train a tokenizer from corpus file.")
    common.add_argument("--vocab-size", type=str, default="500", help="Used with --train-corpus.")
    common.add_argument(
        "--special-token",
        type=str,
        action="append",
        default=[],
        help="Special token string (can be repeated). Treated as atomic during encode.",
    )

    p_encode = subparsers.add_parser("encode", help="Encode a single text into token ids.", parents=[common])
    p_encode.add_argument("--text", type=str, required=True)
    p_encode.add_argument("--format", choices=["json", "space"], default="json")

    p_decode = subparsers.add_parser("decode", help="Decode token ids into text.", parents=[common])
    p_decode.add_argument("token_ids", type=int, nargs="+")

    p_encode_iter = subparsers.add_parser(
        "encode_iterable", help="Encode text from a file as chunks.", parents=[common]
    )
    p_encode_iter.add_argument("--file", type=str, required=True)
    p_encode_iter.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters. If omitted, use lines.")
    p_encode_iter.add_argument("--format", choices=["json", "space"], default="json")

    args = parser.parse_args(argv)
    tokenizer = _build_tokenizer_from_args(args)

    if args.cmd == "encode":
        token_ids = tokenizer.encode(args.text)
        if args.format == "json":
            sys.stdout.write(json.dumps(token_ids))
        else:
            sys.stdout.write(" ".join(str(x) for x in token_ids))
        return 0

    if args.cmd == "decode":
        sys.stdout.write(tokenizer.decode(list(args.token_ids)))
        return 0

    if args.cmd == "encode_iterable":
        path = Path(args.file)
        chunks = _read_chunks_text(path, args.chunk_size)
        token_ids = list(tokenizer.encode_iterable(chunks))
        if args.format == "json":
            sys.stdout.write(json.dumps(token_ids))
        else:
            sys.stdout.write(" ".join(str(x) for x in token_ids))
        return 0

    raise AssertionError(f"Unexpected cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

