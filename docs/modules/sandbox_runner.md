# Sandbox Runner

## 背景和目标
本项目需要经常对各个软件模块进行独立测试，默认的测试方式是使用本机的物理资源，存在资源不隔离和缺少监控的问题。因此希望将测试脚本打包进一个独立的 docker 容器中运行，预先给 docker 容器分配好资源，防止被测试软件模块 OOM 时影响整台机器；同时 docker 容器也更加容易被监控，我可以收集 docker 容器的性能指标以及日志记录，并在运行测试后查看测试表现

## 模块实现

### Docker 容器沙盒
- 镜像构建：将本项目打包成一个 Docker 镜像，特别需要注意镜像文件系统的使用，data 目录下的训练数据量较大，不适合全部打包进镜像内，需要尝试使用引用本机文件的方式访问 data 文件；其余轻量级的代码文件均可放到容器中。
- 容器部署：沙盒容器的使用场景是执行项目下子模块的功能和性能测试，所以容器部署时需要支持用户指定 CPU、内存资源、以及用户指定的启动命令。

### 沙盒监控
沙盒运行时生成的性能数据可以持久化到本地，但是需要存放在名称格式统一、容器之间相互隔离的文件目录内；具体需要支持的监控数据如下：
- 性能指标：容器需要支持 Prometheus 格式的性能指标，并且包含具体的容器资源使用率、容器内线程资源使用率等资源指标
- 日志：记录容器内命令执行时的日志
- Profiling：当运行命令支持 --prof 选项时，持久化 prof 数据

当沙盒运行完毕后，需要支持一个监控方案来查看沙盒运行时的监控数据，你可以使用 Prometheus + Grafana 的方案

## 具体实现（本仓库）

### 沙盒运行方式（CLI）

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

### 数据挂载与镜像构建

- 镜像构建时会通过 `.dockerignore` 排除 `data/`（避免把大数据打进镜像）。
- 沙盒运行时会把宿主机 `data/` 以只读方式挂载到容器：`/workspace/data`。
- `.prof` 目录会挂载到容器：`/workspace/.prof`，用于持久化 profiling 结果（当你的启动命令支持 `--profile/--profile-dir` 或等价 `--prof` 选项时，输出文件会写入该挂载目录）。

### 输出目录结构（隔离持久化）

`--output-root/<run-id>/` 下包含：

- `metrics/metrics.prom`：Prometheus exposition format 的指标文本（容器内会持续更新）
- `logs/command.log`：容器内命令的 stdout/stderr 合并日志
- `prof/`：profiling `.prof` 文件（与容器内 `/workspace/.prof` 同步）

### Prometheus + Grafana

指标会在容器内通过 HTTP 暴露：

- `http://<sandbox-host>:<metrics-host-port>/metrics`

本仓库提供最小监控栈：

- `monitoring/docker-compose.yml`
- `monitoring/prometheus.yml`

你只需要保证 `sandbox_runner_cli.py` 的 `--metrics-host-port` 与 `monitoring/prometheus.yml` 中配置一致（默认目标端口为 `19090`）。