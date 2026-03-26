# Sandbox Runner CLI

本节描述 `cli/sandbox/sandbox_runner_cli.py` 的使用方式：在 docker 沙盒中运行指定命令，并将日志/指标/profiling 数据持久化到本地。

## 运行方式

使用 `sandbox_runner_cli.py` 启动一个资源受限的 docker 容器，并在容器内执行你指定的命令：

```bash
python cli/sandbox/sandbox_runner_cli.py \
  --cpu 1.0 \
  --memory 512m \
  --metrics-host-port 19090 \
  --output-root runs \
  --dry-run \
  -- python -c "print('hello')"
```

- 传入 `--dry-run` 时不会调用 docker，只会生成隔离输出目录并落盘 docker 命令（便于审计与 CI 环境测试）。
- 去掉 `--dry-run` 即会真实构建镜像并运行容器。

## 数据挂载与镜像构建

- 镜像构建时会通过 `.dockerignore` 排除 `data/`（避免把大数据打进镜像）。
- 沙盒运行时会把宿主机 `data/` 以只读方式挂载到容器：`/workspace/data`。
- `.prof` 目录会挂载到容器：`/workspace/.prof`，用于持久化 profiling 结果（当你的启动命令支持 `--profile/--profile-dir` 或等价 `--prof` 选项时，输出文件会写入该挂载目录）。

## 输出目录结构（隔离持久化）

`--output-root/<run-id>/` 下包含：

- `metrics/metrics.prom`：Prometheus exposition format 的指标文本（容器内会持续更新）
- `logs/command.log`：容器内命令的 stdout/stderr 合并日志
- `prof/`：profiling `.prof` 文件（与容器内 `/workspace/.prof` 同步）

## Prometheus + Grafana

指标会在容器内通过 HTTP 暴露：

- `http://<sandbox-host>:<metrics-host-port>/metrics`

本仓库提供最小监控栈：

- `monitoring/docker-compose.yml`
- `monitoring/prometheus.yml`

你只需要保证 `sandbox_runner_cli.py` 的 `--metrics-host-port` 与 `monitoring/prometheus.yml` 中配置一致（默认目标端口为 `19090`）。

