from __future__ import annotations

import dataclasses
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable


@dataclasses.dataclass(frozen=True)
class SandboxRunConfig:
    """
    Sandbox 执行配置。

    约定：
    - 输出隔离目录使用 run_id 作为根目录名
    - 容器内 workspace 固定为 /workspace（便于复用现有 CLI 的相对路径习惯）
    - 挂载 data 到 /workspace/data（只读），挂载 prof 到 /workspace/.prof
    """

    cmd: list[str]
    cpu: float
    memory: str
    run_id: str
    output_root: Path

    project_root: Path
    dockerfile: Path
    image_tag: str

    data_dir: Path

    dry_run: bool = False
    build_image: bool = True

    # 传给 `docker run` 的额外能力。默认包含 PERFMON，便于容器内 `perf trace` 使用 perf_event_open。
    # 若需最小权限，在 CLI 传 --no-sandbox-perf-caps，或在此处显式传入 docker_cap_add=()。
    docker_cap_add: tuple[str, ...] = ("PERFMON",)


@dataclasses.dataclass(frozen=True)
class SandboxRunResult:
    run_dir: Path
    logs_dir: Path
    prof_dir: Path
    docker_build_cmd: list[str]
    docker_run_cmd: list[str]
    exit_code: int | None
    stdout: str
    stderr: str


def _maybe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def prepare_run_dirs(cfg: SandboxRunConfig) -> tuple[Path, Path, Path]:
    run_dir = cfg.output_root / cfg.run_id
    logs_dir = run_dir / "logs"
    prof_dir = run_dir / "prof"

    _maybe_makedirs(logs_dir)
    _maybe_makedirs(prof_dir)

    # 确保日志落盘目录在 dry-run 情况下也能落地，便于脚本联动。
    (logs_dir / "command.log").touch(exist_ok=True)
    (logs_dir / "src.log").touch(exist_ok=True)

    return run_dir, logs_dir, prof_dir


def assemble_docker_build_cmd(cfg: SandboxRunConfig) -> list[str]:
    return [
        "docker",
        "build",
        # 在部分 WSL/公司网络环境下，buildkit 阶段容器可能无法解析 DNS。
        # 使用 host network 可复用宿主机的网络与 DNS 配置，提升构建成功率。
        "--network=host",
        "-t",
        cfg.image_tag,
        "-f",
        str(cfg.dockerfile),
        str(cfg.project_root),
    ]


def assemble_docker_run_cmd(cfg: SandboxRunConfig, run_dir: Path) -> list[str]:
    # 容器内输出目录（挂载 run_dir）
    container_out_dir = "/sandbox_out"
    container_data_dir = "/workspace/data"
    container_prof_dir = "/workspace/.prof"

    container_name = f"llm-sandbox-{cfg.run_id}"

    docker_cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--cpus",
        str(cfg.cpu),
        "--memory",
        cfg.memory,
    ]
    for cap in cfg.docker_cap_add:
        docker_cmd.extend(["--cap-add", cap])
    if "PERFMON" in cfg.docker_cap_add:
        docker_cmd.extend([
            "-v", "/sys/kernel/debug:/sys/kernel/debug:ro",
            "-v", "/sys/kernel/tracing:/sys/kernel/tracing:ro",
        ])
    docker_cmd += [
        "-v",
        f"{str(cfg.data_dir)}:{container_data_dir}:ro",
        "-v",
        f"{str(run_dir)}:{container_out_dir}",
        "-v",
        f"{str(run_dir / 'prof')}:{container_prof_dir}",
        "-e",
        f"SANDBOX_RUN_ID={cfg.run_id}",
        "-e",
        f"SANDBOX_OUT_DIR={container_out_dir}",
        "-e",
        f"SANDBOX_CPU_LIMIT={str(cfg.cpu)}",
        "-e",
        f"SANDBOX_SRC_LOG_PATH={container_out_dir}/logs/src.log",
    ]
    docker_cmd += [
        cfg.image_tag,
        "--out-dir",
        container_out_dir,
        "--",
        *cfg.cmd,
    ]

    return docker_cmd


def _run_streaming(
    cmd: list[str],
    cwd: str,
    log: Callable[[str], None] | None = None,
) -> tuple[int, str, str]:
    """运行子进程并实时流式输出 stdout/stderr，同时收集到字符串中返回。"""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        if log:
            log(line)
        else:
            sys.stdout.write(line)
            sys.stdout.flush()
    rc = proc.wait()
    return rc, "".join(lines), ""


def run_sandbox(
    cfg: SandboxRunConfig,
    log: Callable[[str], None] | None = None,
) -> SandboxRunResult:
    _log = log or (lambda s: (sys.stderr.write(s), sys.stderr.flush()))

    run_dir, logs_dir, prof_dir = prepare_run_dirs(cfg)

    docker_build_cmd = assemble_docker_build_cmd(cfg)
    docker_run_cmd = assemble_docker_run_cmd(cfg, run_dir)

    if cfg.dry_run:
        (run_dir / "docker_commands.txt").write_text(
            "DOCKER_BUILD:\n" + _quote_cmd(docker_build_cmd) + "\n\nDOCKER_RUN:\n" + _quote_cmd(docker_run_cmd) + "\n",
            encoding="utf-8",
        )
        return SandboxRunResult(
            run_dir=run_dir,
            logs_dir=logs_dir,
            prof_dir=prof_dir,
            docker_build_cmd=docker_build_cmd,
            docker_run_cmd=docker_run_cmd,
            exit_code=0,
            stdout="dry-run: docker commands written to docker_commands.txt",
            stderr="",
        )

    if cfg.build_image:
        _log("[sandbox] docker build starting ...\n")
        rc, stdout, stderr = _run_streaming(docker_build_cmd, cwd=str(cfg.project_root), log=log)
        if rc != 0:
            raise RuntimeError(
                "docker build failed:\n"
                f"cmd: {_quote_cmd(docker_build_cmd)}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}\n"
            )
        _log("[sandbox] docker build done.\n")

    _log("[sandbox] docker run starting ...\n")
    rc, stdout, stderr = _run_streaming(docker_run_cmd, cwd=str(cfg.project_root), log=log)
    _log(f"[sandbox] docker run finished (exit_code={rc}).\n")

    return SandboxRunResult(
        run_dir=run_dir,
        logs_dir=logs_dir,
        prof_dir=prof_dir,
        docker_build_cmd=docker_build_cmd,
        docker_run_cmd=docker_run_cmd,
        exit_code=rc,
        stdout=stdout,
        stderr=stderr,
    )


def generate_run_id() -> str:
    """生成默认 run_id（如调用方不传 run_id 时）。"""

    return str(uuid.uuid4())

