from __future__ import annotations

import http.server
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def _read_text_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _parse_kv_lines(text: str) -> dict[str, float]:
    """
    解析类似：
      usage_usec 123
      user_usec 12
    的 cgroup 文件内容。
    """

    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        k, v = parts
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _get_cgroup_relpath_v2() -> str | None:
    """
    cgroup v2：/proc/self/cgroup 中一般形如：
      0::/user.slice/user-1000.slice/session-2.scope
    """

    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except Exception:
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) != 3:
            continue
        _, _, path = parts
        # v2 controller line is path-like: starts with '/'
        if path.startswith("/"):
            return path.lstrip("/")
    return None


def _safe_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _count_threads_in_proc() -> int:
    """
    统计容器内可见进程的 Threads 总数（基于 /proc/*/status）。
    这是“线程资源使用率”的一个可行替代指标。
    """

    proc_root = Path("/proc")
    total = 0

    # 只采样数字 pid 目录，避免扫描其它项
    for pid_dir in proc_root.iterdir():
        name = pid_dir.name
        if not name.isdigit():
            continue
        status_path = pid_dir / "status"
        try:
            # status 文件很小，直接 read_text 即可
            status = status_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in status.splitlines():
            if line.startswith("Threads:"):
                # Threads:      1
                _, val = line.split(":", 1)
                total += int(val.strip())
                break

    return total


def _atomic_write_text(path: str, content: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(out_path)


def _render_prometheus_text(
    *,
    run_id: str,
    cpu_limit: float | None,
    cpu_usage_seconds_total: float | None,
    cpu_usage_rate: float | None,
    memory_current: int | None,
    memory_max: int | None,
    threads_count: int | None,
    pids_current: int | None,
    now_ts: float,
) -> str:
    # 采用稳定的 labels，便于 Grafana 按 run_id 分组查看。
    label = f'run_id="{run_id}"'

    lines: list[str] = []
    lines.append("# TYPE sandbox_cpu_usage_seconds_total counter")
    if cpu_usage_seconds_total is not None:
        lines.append(f"sandbox_cpu_usage_seconds_total{{{label}}} {cpu_usage_seconds_total:.6f}")
    lines.append("# TYPE sandbox_cpu_usage_rate gauge")
    if cpu_usage_rate is not None:
        lines.append(f"sandbox_cpu_usage_rate{{{label}}} {cpu_usage_rate:.6f}")

    lines.append("# TYPE sandbox_memory_bytes gauge")
    if memory_current is not None:
        lines.append(f"sandbox_memory_bytes{{{label}}} {memory_current}")
    lines.append("# TYPE sandbox_memory_max_bytes gauge")
    if memory_max is not None:
        lines.append(f"sandbox_memory_max_bytes{{{label}}} {memory_max}")

    lines.append("# TYPE sandbox_threads_count gauge")
    if threads_count is not None:
        lines.append(f"sandbox_threads_count{{{label}}} {threads_count}")

    lines.append("# TYPE sandbox_pids_current gauge")
    if pids_current is not None:
        lines.append(f"sandbox_pids_current{{{label}}} {pids_current}")

    lines.append("# TYPE sandbox_now_timestamp_seconds gauge")
    lines.append(f"sandbox_now_timestamp_seconds{{{label}}} {now_ts:.6f}")

    if cpu_limit is not None:
        lines.append("# TYPE sandbox_cpu_limit gauge")
        lines.append(f"sandbox_cpu_limit{{{label}}} {cpu_limit}")

    # Prometheus 文本末尾换行通常更友好
    return "\n".join(lines) + "\n"


@dataclass
class MetricsSnapshot:
    cpu_usage_seconds_total: float | None
    cpu_usage_rate: float | None
    memory_current: int | None
    memory_max: int | None
    threads_count: int | None
    pids_current: int | None


class PrometheusMetricsCollector:
    def __init__(
        self,
        *,
        run_id: str,
        out_metrics_path: str,
        interval_s: float,
        metrics_http_port: int,
        cpu_limit: float | None,
    ) -> None:
        self._run_id = run_id
        self._out_metrics_path = out_metrics_path
        self._interval_s = interval_s
        self._cpu_limit = cpu_limit
        self._metrics_http_port = metrics_http_port

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._http_thread: threading.Thread | None = None
        self._httpd: socketserver.TCPServer | None = None

        self._cgroup_relpath_v2 = _get_cgroup_relpath_v2()

        self._prev_usage_usec: float | None = None
        self._prev_t: float | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # 同时启动一个简单 HTTP 服务，用 /metrics 返回当前 metrics 文件内容。
        self._http_thread = threading.Thread(target=self._serve_http, daemon=True)
        self._http_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass

    def _read_snapshot(self) -> MetricsSnapshot:
        now = time.time()

        # ---------- CPU ----------
        cpu_usage_seconds_total: float | None = None
        cpu_usage_rate: float | None = None
        usage_usec: float | None = None
        if self._cgroup_relpath_v2 is not None:
            cpu_stat_path = f"/sys/fs/cgroup/{self._cgroup_relpath_v2}/cpu.stat"
            cpu_stat_text = _read_text_file(cpu_stat_path)
            if cpu_stat_text is not None:
                kv = _parse_kv_lines(cpu_stat_text)
                usage_usec = kv.get("usage_usec")

        if usage_usec is not None:
            cpu_usage_seconds_total = usage_usec / 1e6
            if self._prev_usage_usec is not None and self._prev_t is not None:
                dt = now - self._prev_t
                dus = usage_usec - self._prev_usage_usec
                if dt > 0 and dus >= 0:
                    # usage_usec / 1e6 gives seconds of CPU time across all threads
                    cpu_usage_rate = (dus / 1e6) / dt

            self._prev_usage_usec = usage_usec
            self._prev_t = now

        # ---------- Memory ----------
        memory_current: int | None = None
        memory_max: int | None = None
        if self._cgroup_relpath_v2 is not None:
            mem_current_path = f"/sys/fs/cgroup/{self._cgroup_relpath_v2}/memory.current"
            mem_max_path = f"/sys/fs/cgroup/{self._cgroup_relpath_v2}/memory.max"

            mem_current_text = _read_text_file(mem_current_path)
            memory_current = _safe_int(mem_current_text.strip()) if mem_current_text else None

            mem_max_text = _read_text_file(mem_max_path)
            memory_max: int | None = None
            if mem_max_text is not None:
                mem_max_text = mem_max_text.strip()
                if mem_max_text != "max":
                    memory_max = _safe_int(mem_max_text)

            memory_max = memory_max

        # ---------- Threads ----------
        threads_count: int | None = None
        try:
            threads_count = _count_threads_in_proc()
        except Exception:
            threads_count = None

        # ---------- Pids (optional) ----------
        pids_current: int | None = None
        if self._cgroup_relpath_v2 is not None:
            pids_current_path = f"/sys/fs/cgroup/{self._cgroup_relpath_v2}/pids.current"
            pids_current = _safe_int(_read_text_file(pids_current_path))

        return MetricsSnapshot(
            cpu_usage_seconds_total=cpu_usage_seconds_total,
            cpu_usage_rate=cpu_usage_rate,
            memory_current=memory_current,
            memory_max=memory_max,
            threads_count=threads_count,
            pids_current=pids_current,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._read_snapshot()
            content = _render_prometheus_text(
                run_id=self._run_id,
                cpu_limit=self._cpu_limit,
                cpu_usage_seconds_total=snapshot.cpu_usage_seconds_total,
                cpu_usage_rate=snapshot.cpu_usage_rate,
                memory_current=snapshot.memory_current,
                memory_max=snapshot.memory_max,
                threads_count=snapshot.threads_count,
                pids_current=snapshot.pids_current,
                now_ts=time.time(),
            )
            try:
                _atomic_write_text(self._out_metrics_path, content)
            except Exception:
                # 监控不应影响主命令运行
                pass

            # interval 用 stop_event.wait，避免 stop 时还要 sleep 完整周期
            self._stop_event.wait(self._interval_s)

    def _serve_http(self) -> None:
        collector = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    metrics = Path(collector._out_metrics_path).read_text(encoding="utf-8")
                except Exception:
                    metrics = ""
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(metrics.encode("utf-8"))

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                # 避免刷屏
                return

        with socketserver.TCPServer(("", self._metrics_http_port), Handler) as httpd:
            self._httpd = httpd
            httpd.serve_forever()

