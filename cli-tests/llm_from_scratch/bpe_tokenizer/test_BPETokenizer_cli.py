from __future__ import annotations

import json
from pathlib import Path

SCRIPT = "cli/llm_from_scratch/bpe_tokenizer/BPETokenizer_cli.py"


def test_help(run_cli) -> None:
    r = run_cli(SCRIPT, ["--help"])
    assert r.returncode == 0
    assert "BPETokenizer" in r.stdout


def test_pretokenize_text(run_cli, tmp_path: Path) -> None:
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "pretokenize",
            "--text",
            "hi",
            "--vocab-version",
            "901",
            "--force-rebuild",
            "--format",
            "json",
        ],
        env=env,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert all(isinstance(x, list) for x in data)


def test_pretokenize_neither_text_nor_file(run_cli, tmp_path: Path) -> None:
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        ["pretokenize", "--vocab-version", "902", "--force-rebuild"],
        env=env,
    )
    assert r.returncode == 1


def test_vocab_info_force_rebuild(run_cli, tmp_path: Path) -> None:
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        ["vocab-info", "--vocab-version", "903", "--force-rebuild"],
        env=env,
    )
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "257"
    assert lines[1] == "256"


def test_pretokenize_from_file(run_cli, tmp_path: Path) -> None:
    txt = tmp_path / "in.txt"
    txt.write_text("a b", encoding="utf-8")
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "pretokenize",
            "--file",
            str(txt),
            "--vocab-version",
            "904",
            "--force-rebuild",
            "--format",
            "json",
        ],
        env=env,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)
