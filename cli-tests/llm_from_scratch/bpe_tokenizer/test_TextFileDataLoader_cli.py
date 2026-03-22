from __future__ import annotations

from pathlib import Path

SCRIPT = "cli/llm_from_scratch/bpe_tokenizer/TextFileDataLoader_cli.py"


def test_help(run_cli, project_root: Path) -> None:
    r = run_cli(SCRIPT, ["--help"])
    assert r.returncode == 0
    assert "TextFileDataLoader" in r.stdout


def test_info_and_read(tmp_path: Path, run_cli) -> None:
    f = tmp_path / "t.txt"
    f.write_bytes(b"hello")
    r = run_cli(SCRIPT, ["info", "--file", str(f)])
    assert r.returncode == 0
    assert r.stdout.strip() == "5"

    r2 = run_cli(SCRIPT, ["read", "--file", str(f), "--size", "2"])
    assert r2.returncode == 0
    assert r2.stdout == "he"

    r3 = run_cli(SCRIPT, ["read", "--file", str(f), "--size", "2", "--offset", "2"])
    assert r3.returncode == 0
    assert r3.stdout == "ll"


def test_read_invalid_size(tmp_path: Path, run_cli) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    r = run_cli(SCRIPT, ["read", "--file", str(f), "--size", "0"])
    assert r.returncode == 1
    assert "error" in r.stderr.lower() or "正整数" in r.stderr


def test_missing_file(run_cli) -> None:
    r = run_cli(SCRIPT, ["info", "--file", "/nonexistent/___missing___"])
    assert r.returncode == 1


def test_cursor_set_negative(tmp_path: Path, run_cli) -> None:
    f = tmp_path / "x.txt"
    f.write_text("ab")
    r = run_cli(SCRIPT, ["cursor-set", "--file", str(f), "--position", "-1"])
    assert r.returncode == 1


def test_cursor_get_and_reset(tmp_path: Path, run_cli) -> None:
    f = tmp_path / "x.txt"
    f.write_text("xyz")
    r = run_cli(SCRIPT, ["cursor-get", "--file", str(f)])
    assert r.returncode == 0
    assert r.stdout.strip() == "0"

    r2 = run_cli(SCRIPT, ["reset", "--file", str(f)])
    assert r2.returncode == 0
    assert r2.stdout.strip() == "0"


def test_read_print_cursor(tmp_path: Path, run_cli) -> None:
    f = tmp_path / "x.txt"
    f.write_text("abc")
    r = run_cli(SCRIPT, ["read", "--file", str(f), "--size", "2", "--print-cursor"])
    assert r.returncode == 0
    assert r.stdout == "ab"
    assert r.stderr.strip().endswith("2")
