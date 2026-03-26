from __future__ import annotations

import dataclasses
import shlex
import subprocess
import uuid
from pathlib import Path


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
    metrics_host_port: int

    dry_run: bool = False
    build_image: bool = True
    metrics_interval_s: float = 1.0

    # “你可以使用 Prometheus + Grafana”的前提是指标采用 Prometheus exposition format。
    # 这里容器内指标服务端口固定，host 通过 -p 映射到 metrics_host_port。
    container_metrics_port: int = 8000


@dataclasses.dataclass(frozen=True)
class SandboxRunResult:
    run_dir: Path
    metrics_dir: Path
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


def prepare_run_dirs(cfg: SandboxRunConfig) -> tuple[Path, Path, Path, Path]:
    run_dir = cfg.output_root / cfg.run_id
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    prof_dir = run_dir / "prof"

    _maybe_makedirs(metrics_dir)
    _maybe_makedirs(logs_dir)
    _maybe_makedirs(prof_dir)

    # 确保“日志/指标落盘目录”在 dry-run 情况下也能落地，便于脚本联动。
    (logs_dir / "command.log").touch(exist_ok=True)
    (metrics_dir / "metrics.prom").touch(exist_ok=True)

    return run_dir, metrics_dir, logs_dir, prof_dir


def assemble_docker_build_cmd(cfg: SandboxRunConfig) -> list[str]:
    return [
        "docker",
        "build",
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

    # 容器内 metrics 文本落盘位置（collector 会写这个文件）
    container_metrics_path = f"{container_out_dir}/metrics/metrics.prom"

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
        "-p",
        f"{cfg.metrics_host_port}:{cfg.container_metrics_port}",
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
        f"SANDBOX_METRICS_PATH={container_metrics_path}",
        "-e",
        f"SANDBOX_METRICS_INTERVAL_S={str(cfg.metrics_interval_s)}",
        "-e",
        f"SANDBOX_CPU_LIMIT={str(cfg.cpu)}",
        cfg.image_tag,
        "--out-dir",
        container_out_dir,
        "--metrics-port",
        str(cfg.container_metrics_port),
        "--metrics-path",
        container_metrics_path,
        "--metrics-interval-s",
        str(cfg.metrics_interval_s),
        "--",
        *cfg.cmd,
    ]

    return docker_cmd


def run_sandbox(cfg: SandboxRunConfig) -> SandboxRunResult:
    run_dir, metrics_dir, logs_dir, prof_dir = prepare_run_dirs(cfg)

    docker_build_cmd = assemble_docker_build_cmd(cfg)
    docker_run_cmd = assemble_docker_run_cmd(cfg, run_dir)

    if cfg.dry_run:
        (run_dir / "docker_commands.txt").write_text(
            "DOCKER_BUILD:\n" + _quote_cmd(docker_build_cmd) + "\n\nDOCKER_RUN:\n" + _quote_cmd(docker_run_cmd) + "\n",
            encoding="utf-8",
        )
        return SandboxRunResult(
            run_dir=run_dir,
            metrics_dir=metrics_dir,
            logs_dir=logs_dir,
            prof_dir=prof_dir,
            docker_build_cmd=docker_build_cmd,
            docker_run_cmd=docker_run_cmd,
            exit_code=0,
            stdout="dry-run: docker commands written to docker_commands.txt",
            stderr="",
        )

    if cfg.build_image:
        proc_build = subprocess.run(
            docker_build_cmd,
            cwd=str(cfg.project_root),
            capture_output=True,
            text=True,
        )
        if proc_build.returncode != 0:
            raise RuntimeError(
                "docker build failed:\n"
                f"cmd: {_quote_cmd(docker_build_cmd)}\n"
                f"stdout:\n{proc_build.stdout}\n"
                f"stderr:\n{proc_build.stderr}\n"
            )

    proc_run = subprocess.run(
        docker_run_cmd,
        cwd=str(cfg.project_root),
        capture_output=True,
        text=True,
    )

    return SandboxRunResult(
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        logs_dir=logs_dir,
        prof_dir=prof_dir,
        docker_build_cmd=docker_build_cmd,
        docker_run_cmd=docker_run_cmd,
        exit_code=proc_run.returncode,
        stdout=proc_run.stdout,
        stderr=proc_run.stderr,
    )


def generate_run_id() -> str:
    """生成默认 run_id（如调用方不传 run_id 时）。"""

    return str(uuid.uuid4())

