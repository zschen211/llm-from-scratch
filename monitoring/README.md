# Sandbox Monitoring (Prometheus + Grafana)

沙盒 runner 的容器内会启动一个简易 HTTP 服务，把指标按 Prometheus exposition format 暴露为：

- `http://<sandbox-host>:<metrics-host-port>/metrics`

指标还会落盘到本地目录 `--output-root/<run-id>/metrics/metrics.prom`，因此你也可以在沙盒结束后直接离线查看。

## 快速开始

> 说明：若你的环境无法直连 Docker Hub，本仓库已在 `monitoring/docker-compose.yml` 中将镜像地址切换为国内镜像代理前缀（`docker.m.daocloud.io/...`）。

1. 启动沙盒（示例）：

   ```bash
   python cli/sandbox/sandbox_runner_cli.py \
     --cpu 1.0 --memory 512m \
     --metrics-host-port 19090 \
     --dry-run \
     -- python -c "print('hello')"
   ```

   注：去掉 `--dry-run` 才会真正调用 docker。

2. 启动监控栈（另开终端）：

   ```bash
   cd monitoring
   sudo docker compose up -d
   ```

3. 打开 Grafana：

   - URL: http://localhost:3000
   - Prometheus: http://localhost:9090

## 端口对齐

`monitoring/prometheus.yml` 默认 scrape 的目标是 `19090`（同时尝试 `host.docker.internal` 与 `localhost`）。
因此你需要让 sandbox runner 的 `--metrics-host-port` 与这里一致，或自行修改 `prometheus.yml`。

