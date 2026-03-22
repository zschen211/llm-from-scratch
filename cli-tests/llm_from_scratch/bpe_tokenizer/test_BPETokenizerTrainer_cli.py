from __future__ import annotations

SCRIPT = "cli/llm_from_scratch/bpe_tokenizer/BPETokenizerTrainer_cli.py"


def test_help(run_cli) -> None:
    r = run_cli(SCRIPT, ["--help"])
    assert r.returncode == 0
    assert "BPETokenizerTrainer" in r.stdout


def test_len(run_cli, tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_bytes(b"abcd")
    r = run_cli(SCRIPT, ["len", "--file", str(f)])
    assert r.returncode == 0
    assert r.stdout.strip() == "4"


def test_preview(run_cli, tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("Once upon", encoding="utf-8")
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "preview",
            "--file",
            str(f),
            "--chunk-size",
            "100",
            "--vocab-version",
            "905",
            "--force-rebuild",
            "--head",
            "2",
        ],
        env=env,
    )
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert int(lines[0]) > 0
    assert int(lines[1]) > 0


def test_cursor_flow(run_cli, tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("abcdef", encoding="utf-8")
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "cursor-flow",
            "--file",
            str(f),
            "--read-size",
            "2",
            "--vocab-version",
            "906",
        ],
        env=env,
    )
    assert r.returncode == 0
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    assert lines == ["0", "2", "0"]


def test_parallel_multiprocess(run_cli, tmp_path: Path) -> None:
    f = tmp_path / "dummy.txt"
    f.write_text("x", encoding="utf-8")
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "parallel",
            "--file",
            str(f),
            "--inputs",
            "1",
            "2",
            "3",
            "--n-processes",
            "2",
            "--vocab-version",
            "907",
            "--force-rebuild",
        ],
        env=env,
    )
    assert r.returncode == 0
    assert "[2, 3, 4]" in r.stdout.replace(" ", "") or "[2,3,4]" in r.stdout.replace(" ", "")


def test_parallel_single_process(run_cli, tmp_path: Path) -> None:
    f = tmp_path / "d.txt"
    f.write_text("z", encoding="utf-8")
    env = {"LLM_FS_TOKENIZER_DIR": str(tmp_path)}
    r = run_cli(
        SCRIPT,
        [
            "parallel",
            "--file",
            str(f),
            "--inputs",
            "10",
            "--n-processes",
            "1",
            "--vocab-version",
            "908",
            "--force-rebuild",
        ],
        env=env,
    )
    assert r.returncode == 0
    assert "11" in r.stdout


def test_missing_file(run_cli) -> None:
    r = run_cli(SCRIPT, ["len", "--file", "/no/such/file/xxx"])
    assert r.returncode == 1
