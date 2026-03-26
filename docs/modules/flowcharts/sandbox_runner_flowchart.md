# `sandbox_runner` 关键流程（容器隔离 + 指标采集）

对应代码：
- `src/sandbox_runner.py`：宿主机侧构建/运行容器，并准备挂载目录
- `src/sandbox_runner_entrypoint.py`：容器内入口（启动指标采集 + 执行用户命令并落日志）
- `src/sandbox_runner_metrics.py`：cgroup 采样并以 Prometheus `exposition` 格式输出（同时提供 `/metrics` HTTP）

```mermaid
flowchart TD
  subgraph HOST["宿主机 src/sandbox_runner.py"]
    H0[配置运行参数]
    H1[创建输出目录]
    H2[docker build]
    H3[docker run 资源限制和挂载]
    H4[收集输出和退出码]
    H0 --> H1 --> H2 --> H3 --> H4
  end

  subgraph CONTAINER["容器入口 src/sandbox_runner_entrypoint.py"]
    C0[解析参数和用户命令]
    C1[启动指标采集]
    C2[在 workspace 执行用户命令 并写日志]
    C3[停止采集 并写 exit code]
    C0 --> C1 --> C2 --> C3
  end

  subgraph METRICS["指标采集 src/sandbox_runner_metrics.py"]
    M0[循环采样 cgroup 和 proc]
    M1[渲染 Prometheus 指标文本]
    M2[写 metrics 文件]
    M3[提供 HTTP /metrics]
    M0 --> M1 --> M2 --> M3
  end

  HOST --> CONTAINER
  CONTAINER --> METRICS
```

