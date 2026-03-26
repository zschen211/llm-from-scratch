from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.bpe_tokenizer import train_bpe  # noqa: E402


def _load_checkpoint(path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = {int(k): bytes(v) for k, v in payload["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in payload["merges"]]
    return vocab, merges


def test_train_bpe_cli_checkpoint_matches_api(run_cli, project_root: Path, tmp_path: Path) -> None:
    input_path = project_root / "tests" / "fixtures" / "corpus.en"
    vocab_size = 500
    special = "<|endoftext|>"

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=[special],
    )

    out = tmp_path / "out.json"

    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py",
        [
            "--input-corpus",
            str(input_path),
            "--vocab-size",
            str(vocab_size),
            "--special-token",
            special,
            "--out",
            str(out),
            "--no-print-metrics",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()

    vocab2, merges2 = _load_checkpoint(out)
    assert vocab2 == vocab
    assert merges2 == merges

