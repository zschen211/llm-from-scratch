from __future__ import annotations

from pathlib import Path


def test_sandbox_runner_cli_help(run_cli, project_root: Path) -> None:
    proc = run_cli(
        "cli/sandbox/sandbox_runner_cli.py",
        ["--help"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "sandbox runner" in proc.stdout.lower()


def test_sandbox_runner_cli_dry_run_writes_run_dir(run_cli, tmp_path: Path) -> None:
    run_id = "test-sandbox-run-1"
    proc = run_cli(
        "cli/sandbox/sandbox_runner_cli.py",
        [
            "--dry-run",
            "--cpu",
            "1.5",
            "--memory",
            "256m",
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path),
            "--metrics-host-port",
            "19123",
            "--skip-build",
            "--",
            "python",
            "-c",
            "print('hello from sandbox')",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert f"run_id: {run_id}" in proc.stdout

    run_dir = tmp_path / run_id
    assert run_dir.is_dir()
    assert (run_dir / "docker_commands.txt").is_file()
    assert (run_dir / "logs" / "command.log").is_file()
    assert (run_dir / "metrics" / "metrics.prom").is_file()

    assert "docker_run_cmd:" in proc.stdout
    assert "docker_build_cmd:" in proc.stdout

