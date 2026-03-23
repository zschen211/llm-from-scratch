from __future__ import annotations

import cProfile
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))


def _work_a() -> int:
    return sum(range(5_000))


def _work_b() -> int:
    return sum(i * i for i in range(5_000))


def _work_top() -> int:
    return _work_a() + _work_b()


def _make_small_prof(path: Path) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        _work_top()
    profiler.disable()
    profiler.dump_stats(str(path))


# --------------- train_bpe --profile tests ---------------


def test_train_bpe_cli_profile_generates_prof_file(run_cli, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world hello world foo bar", encoding="utf-8")
    out = tmp_path / "out.json"
    prof_dir = tmp_path / "profiles"

    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py",
        [
            "--input-corpus",
            str(corpus),
            "--vocab-size",
            "260",
            "--special-token",
            "<|endoftext|>",
            "--out",
            str(out),
            "--profile",
            "--profile-dir",
            str(prof_dir),
            "--no-print-metrics",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert prof_dir.is_dir()
    prof_files = list(prof_dir.glob("*.prof"))
    assert len(prof_files) == 1
    assert "[profile] saved:" in proc.stdout


def test_train_bpe_cli_profile_default_dir(run_cli, project_root: Path, tmp_path: Path) -> None:
    """--profile without --profile-dir defaults to <project_root>/.prof/"""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world", encoding="utf-8")
    out = tmp_path / "out.json"

    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py",
        [
            "--input-corpus",
            str(corpus),
            "--vocab-size",
            "258",
            "--special-token",
            "<|endoftext|>",
            "--out",
            str(out),
            "--profile",
            "--no-print-metrics",
        ],
    )
    assert proc.returncode == 0, proc.stderr

    default_prof_dir = project_root / ".prof"
    if default_prof_dir.is_dir():
        prof_files = list(default_prof_dir.glob("train_bpe_*.prof"))
        assert len(prof_files) >= 1
        for f in prof_files:
            f.unlink()


# --------------- flamegraph CLI tests ---------------


def test_flamegraph_cli_generates_html(run_cli, tmp_path: Path) -> None:
    prof_path = tmp_path / "test.prof"
    _make_small_prof(prof_path)

    html_path = tmp_path / "test.html"
    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py",
        [str(prof_path), "--out", str(html_path)],
    )
    assert proc.returncode == 0, proc.stderr
    assert html_path.is_file()
    content = html_path.read_text(encoding="utf-8")
    assert "d3-flamegraph" in content
    assert "Flame Graph" in content


def test_flamegraph_cli_default_output_name(run_cli, tmp_path: Path) -> None:
    prof_path = tmp_path / "sample.prof"
    _make_small_prof(prof_path)

    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py",
        [str(prof_path)],
    )
    assert proc.returncode == 0, proc.stderr

    expected_html = tmp_path / "sample.html"
    assert expected_html.is_file()


def test_flamegraph_cli_missing_file(run_cli, tmp_path: Path) -> None:
    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py",
        [str(tmp_path / "nonexistent.prof")],
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr


def test_flamegraph_cli_help(run_cli) -> None:
    proc = run_cli(
        "cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py",
        ["--help"],
    )
    assert proc.returncode == 0
    assert "flame graph" in proc.stdout.lower()
