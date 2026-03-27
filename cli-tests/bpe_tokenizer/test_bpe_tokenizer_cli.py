from __future__ import annotations

import json
import sys
from pathlib import Path

# pytest 在收集测试时可能把 `cli-tests` 加入 `sys.path`；测试目录不得放在
# `cli-tests/llm_from_scratch/`（会与可编辑安装的 `llm_from_scratch` 发生命名空间合并），
# 故实际路径为 `cli-tests/cli/llm_from_scratch/...`。
# 仍将项目根目录插到 sys.path 最前面，确保导入落到真实实现。
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.bpe_tokenizer import make_tokenizer, train_bpe


def _save_checkpoint(path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    payload = {
        "vocab": {str(k): list(v) for k, v in vocab.items()},
        "merges": [[list(a), list(b)] for a, b in merges],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bpe_tokenizer_cli_encode_decode(run_cli, project_root: Path, tmp_path: Path) -> None:
    input_path = project_root / "tests" / "fixtures" / "corpus.en"
    vocab_size = 500
    special = "<|endoftext|>"

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=[special],
    )
    tokenizer = make_tokenizer(vocab=vocab, merges=merges, special_tokens=[special])

    ckpt = tmp_path / "tok.json"
    _save_checkpoint(ckpt, vocab, merges)

    text = f"hello {special} world"

    proc = run_cli(
        "cli/bpe_tokenizer/bpe_tokenizer_cli.py",
        [
            "encode",
            "--checkpoint",
            str(ckpt),
            "--special-token",
            special,
            "--format",
            "json",
            "--text",
            text,
        ],
    )
    assert proc.returncode == 0, proc.stderr
    token_ids = json.loads(proc.stdout)
    assert token_ids == tokenizer.encode(text)

    proc = run_cli(
        "cli/bpe_tokenizer/bpe_tokenizer_cli.py",
        [
            "decode",
            "--checkpoint",
            str(ckpt),
            "--special-token",
            special,
            *[str(x) for x in token_ids],
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == tokenizer.decode(token_ids)


def test_bpe_tokenizer_cli_encode_iterable_chunk_size(
    run_cli, project_root: Path, tmp_path: Path
) -> None:
    input_path = project_root / "tests" / "fixtures" / "corpus.en"
    special = "<|endoftext|>"

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=[special],
    )
    tokenizer = make_tokenizer(vocab=vocab, merges=merges, special_tokens=[special])

    ckpt = tmp_path / "tok.json"
    _save_checkpoint(ckpt, vocab, merges)

    # 让 special token 位于文本开头，确保 BPETokenizer.encode_iterable 会按原子方式匹配
    # （避免正则片段在中间吞掉 special，从而与 encode 的行为不一致）。
    text = f"{special} BB"
    fpath = tmp_path / "in.txt"
    fpath.write_text(text, encoding="utf-8")

    proc = run_cli(
        "cli/bpe_tokenizer/bpe_tokenizer_cli.py",
        [
            "encode_iterable",
            "--checkpoint",
            str(ckpt),
            "--special-token",
            special,
            "--file",
            str(fpath),
            "--format",
            "json",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    token_ids = json.loads(proc.stdout)
    assert token_ids == tokenizer.encode(text)

