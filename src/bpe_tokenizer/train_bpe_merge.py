"""BPE 训练：字节对统计、合并迭代、checkpoint 与流式 chunk 读写。"""

from __future__ import annotations

import json
import logging
import multiprocessing as _mp
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

from ._gpt2_bytes import gpt2_byte_positions

# 由模块末尾 try/merge 初始化，避免与 _rust_bridge 循环导入时未定义
RUST_AVAILABLE: bool = False
_rust_count_pairs_fn: Callable[..., dict[tuple[bytes, bytes], int]] | None = None
_rust_merge_pair_all_words_with_deltas_fn: Callable[..., dict[tuple[bytes, bytes], int]] | None = None


def _build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    idx = 0
    for st in special_tokens:
        vocab[idx] = st.encode("utf-8")
        idx += 1
    for b in gpt2_byte_positions():
        vocab[idx] = bytes([b])
        idx += 1
    return vocab


def _count_pairs(words: list[list[bytes]], min_freq: int = 1) -> Counter[tuple[bytes, bytes]]:
    if RUST_AVAILABLE and _rust_count_pairs_fn is not None:
        result = _rust_count_pairs_fn(words, min_freq)
        return Counter(result)

    c: Counter[tuple[bytes, bytes]] = Counter()
    for w in words:
        for i in range(len(w) - 1):
            c[(w[i], w[i + 1])] += 1

    if min_freq > 1:
        c = Counter({p: cnt for p, cnt in c.items() if cnt >= min_freq})

    return c


def _merge_pair_in_word(word: list[bytes], left: bytes, right: bytes, merged: bytes) -> list[bytes]:
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _merge_pair_all_words(words: list[list[bytes]], left: bytes, right: bytes, merged: bytes) -> None:
    for j in range(len(words)):
        words[j] = _merge_pair_in_word(words[j], left, right, merged)


def _merge_pair_all_words_with_pair_deltas(
    words: list[list[bytes]],
    left: bytes,
    right: bytes,
    merged: bytes,
) -> dict[tuple[bytes, bytes], int]:
    if RUST_AVAILABLE and _rust_merge_pair_all_words_with_deltas_fn is not None:
        return _rust_merge_pair_all_words_with_deltas_fn(words, left, right, merged)

    delta: dict[tuple[bytes, bytes], int] = {}
    words_len = len(words)

    for j in range(words_len):
        word = words[j]
        word_len = len(word)
        if word_len < 2:
            continue

        out: list[bytes] = []
        created_merge_pos: list[int] = []
        removed_old_pair_indices: list[int] = []

        k = 0
        did_merge = False
        while k < word_len:
            k_plus_1 = k + 1
            if k_plus_1 < word_len and word[k] == left and word[k_plus_1] == right:
                did_merge = True
                if k > 0:
                    removed_old_pair_indices.append(k - 1)
                removed_old_pair_indices.append(k)
                k_plus_2 = k + 2
                if k_plus_2 < word_len:
                    removed_old_pair_indices.append(k_plus_1)

                created_merge_pos.append(len(out))
                out.append(merged)
                k = k_plus_2
            else:
                out.append(word[k])
                k += 1

        if not did_merge:
            continue

        added_new_pair_positions: set[int] = set()
        out_len = len(out)
        out_len_minus_1 = out_len - 1
        for pos in created_merge_pos:
            if pos > 0:
                added_new_pair_positions.add(pos - 1)
            if pos < out_len_minus_1:
                added_new_pair_positions.add(pos)

        for i in set(removed_old_pair_indices):
            p = (word[i], word[i + 1])
            delta[p] = delta.get(p, 0) - 1

        for i in added_new_pair_positions:
            p = (out[i], out[i + 1])
            delta[p] = delta.get(p, 0) + 1

        words[j] = out

    if delta:
        delta = {p: v for p, v in delta.items() if v != 0}
    return delta


def _pick_pair_to_merge(pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    best_freq = max(pair_counts.values())
    candidates = [p for p, f in pair_counts.items() if f == best_freq]
    return max(candidates)


def _worker_main(
    chunk: list[list[bytes]],
    cmd_q: _mp.Queue,  # type: ignore[type-arg]
    result_q: _mp.Queue,  # type: ignore[type-arg]
) -> None:
    while True:
        cmd = cmd_q.get()
        if cmd == "count":
            t0 = time.perf_counter()
            c = dict(_count_pairs(chunk, min_freq=1))
            elapsed_s = time.perf_counter() - t0
            result_q.put(("count", c, elapsed_s))
        elif isinstance(cmd, tuple) and cmd[0] == "merge_delta":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]

            t0 = time.perf_counter()
            delta = _merge_pair_all_words_with_pair_deltas(chunk, left, right, merged)
            elapsed_s = time.perf_counter() - t0
            result_q.put(("merge_delta", delta, elapsed_s))
        elif isinstance(cmd, tuple) and cmd[0] == "merge":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]
            t0 = time.perf_counter()
            _merge_pair_all_words_with_pair_deltas(chunk, left, right, merged)
            elapsed_s = time.perf_counter() - t0
            result_q.put(("merge", elapsed_s))
        elif cmd == "stop":
            break


class _BPEWorkerPool:
    """内存模式的 worker pool，每个 worker 持有一段 words 切片。"""
    def __init__(self, words: list[list[bytes]], num_workers: int) -> None:
        n = len(words)
        chunk_size = max(1, (n + num_workers - 1) // num_workers)
        self._workers: list[tuple[_mp.Process, _mp.Queue, _mp.Queue]] = []  # type: ignore[type-arg]
        for i in range(num_workers):
            start = i * chunk_size
            if start >= n:
                break
            end = min(start + chunk_size, n)
            chunk = words[start:end]
            cmd_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
            result_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
            p = _mp.Process(target=_worker_main, args=(chunk, cmd_q, result_q), daemon=True)
            p.start()
            self._workers.append((p, cmd_q, result_q))

    def _worker_count(self) -> int:
        return len(self._workers)

    @property
    def alive(self) -> bool:
        return all(p.is_alive() for p, _, _ in self._workers)

    def count_pairs(self) -> tuple[Counter[tuple[bytes, bytes]], dict[str, Any]]:
        worker_n = self._worker_count()
        t_enq0 = time.perf_counter()
        for _, cmd_q, _ in self._workers:
            cmd_q.put("count")
        t_enq1 = time.perf_counter()

        total: Counter[tuple[bytes, bytes]] = Counter()
        t_deq0 = time.perf_counter()
        consumer_task_times_s: list[float] = []
        for _, _, result_q in self._workers:
            _kind, partial, elapsed_s = result_q.get()
            consumer_task_times_s.append(float(elapsed_s))
            for pair, freq in partial.items():
                total[pair] += freq
        t_deq1 = time.perf_counter()

        enqueue_dur_s = max(1e-12, t_enq1 - t_enq0)
        dequeue_dur_s = max(1e-12, t_deq1 - t_deq0)
        consumer_avg_task_ms = 1000.0 * (sum(consumer_task_times_s) / max(1, len(consumer_task_times_s)))

        metrics = {
            "tasks_enqueued": worker_n,
            "tasks_dequeued": worker_n,
            "enqueue_ms": 1000.0 * (t_enq1 - t_enq0),
            "dequeue_wait_ms": 1000.0 * (t_deq1 - t_deq0),
            "enqueue_throughput_tps": worker_n / enqueue_dur_s,
            "dequeue_throughput_tps": worker_n / dequeue_dur_s,
            "consumer_avg_task_ms": consumer_avg_task_ms,
        }
        return total, metrics

    def merge_pair(self, left: bytes, right: bytes, merged: bytes) -> dict[str, Any]:
        worker_n = self._worker_count()
        t_enq0 = time.perf_counter()
        cmd = ("merge", left, right, merged)
        for _, cmd_q, _ in self._workers:
            cmd_q.put(cmd)
        t_enq1 = time.perf_counter()

        consumer_task_times_s: list[float] = []
        t_deq0 = time.perf_counter()
        for _, _, result_q in self._workers:
            _kind, elapsed_s = result_q.get()
            consumer_task_times_s.append(float(elapsed_s))
        t_deq1 = time.perf_counter()

        enqueue_dur_s = max(1e-12, t_enq1 - t_enq0)
        dequeue_dur_s = max(1e-12, t_deq1 - t_deq0)
        consumer_avg_task_ms = 1000.0 * (sum(consumer_task_times_s) / max(1, len(consumer_task_times_s)))

        metrics = {
            "tasks_enqueued": worker_n,
            "tasks_dequeued": worker_n,
            "enqueue_ms": 1000.0 * (t_enq1 - t_enq0),
            "dequeue_wait_ms": 1000.0 * (t_deq1 - t_deq0),
            "enqueue_throughput_tps": worker_n / enqueue_dur_s,
            "dequeue_throughput_tps": worker_n / dequeue_dur_s,
            "consumer_avg_task_ms": consumer_avg_task_ms,
        }
        return metrics

    def merge_pair_with_pair_deltas(
        self, left: bytes, right: bytes, merged: bytes
    ) -> tuple[dict[tuple[bytes, bytes], int], dict[str, Any]]:
        worker_n = self._worker_count()
        t_enq0 = time.perf_counter()
        cmd = ("merge_delta", left, right, merged)
        for _, cmd_q, _ in self._workers:
            cmd_q.put(cmd)
        t_enq1 = time.perf_counter()

        total_delta: dict[tuple[bytes, bytes], int] = {}
        consumer_task_times_s: list[float] = []
        t_deq0 = time.perf_counter()

        for _, _, result_q in self._workers:
            _kind, partial_delta, elapsed_s = result_q.get()
            consumer_task_times_s.append(float(elapsed_s))
            for pair, d in partial_delta.items():
                total_delta[pair] = total_delta.get(pair, 0) + int(d)
        t_deq1 = time.perf_counter()

        if total_delta:
            total_delta = {p: v for p, v in total_delta.items() if v != 0}

        enqueue_dur_s = max(1e-12, t_enq1 - t_enq0)
        dequeue_dur_s = max(1e-12, t_deq1 - t_deq0)
        consumer_avg_task_ms = 1000.0 * (sum(consumer_task_times_s) / max(1, len(consumer_task_times_s)))

        metrics = {
            "tasks_enqueued": worker_n,
            "tasks_dequeued": worker_n,
            "enqueue_ms": 1000.0 * (t_enq1 - t_enq0),
            "dequeue_wait_ms": 1000.0 * (t_deq1 - t_deq0),
            "enqueue_throughput_tps": worker_n / enqueue_dur_s,
            "dequeue_throughput_tps": worker_n / dequeue_dur_s,
            "consumer_avg_task_ms": consumer_avg_task_ms,
        }
        return total_delta, metrics

    def shutdown(self) -> None:
        for _, cmd_q, _ in self._workers:
            try:
                cmd_q.put("stop")
            except Exception:
                pass
        for p, _, _ in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


def _stream_worker_main(
    chunk_files: list[Path],
    use_inverted_index: bool,
    cmd_q: _mp.Queue,  # type: ignore[type-arg]
    result_q: _mp.Queue,  # type: ignore[type-arg]
) -> None:
    """流式模式的 worker，负责处理分配给它的 chunk 文件。"""
    while True:
        cmd = cmd_q.get()
        if cmd == "count":
            t0 = time.perf_counter()
            total_counts: dict[tuple[bytes, bytes], int] = {}

            for chunk_path in chunk_files:
                if use_inverted_index:
                    chunk = _load_words_chunk_with_index(chunk_path)
                    chunk_counts = dict(_count_pairs(chunk.words, min_freq=1))
                else:
                    chunk = _load_words_chunk(chunk_path)
                    chunk_counts = dict(_count_pairs(chunk, min_freq=1))

                for pair, cnt in chunk_counts.items():
                    total_counts[pair] = total_counts.get(pair, 0) + cnt

            elapsed_s = time.perf_counter() - t0
            result_q.put(("count", total_counts, elapsed_s))

        elif isinstance(cmd, tuple) and cmd[0] == "merge_delta":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]

            t0 = time.perf_counter()
            total_delta: dict[tuple[bytes, bytes], int] = {}

            for chunk_path in chunk_files:
                if use_inverted_index:
                    chunk = _load_words_chunk_with_index(chunk_path)
                    delta = chunk.merge_pair_with_deltas(left, right, merged)
                    _dump_words_chunk_with_index(chunk_path, chunk)
                else:
                    chunk = _load_words_chunk(chunk_path)
                    delta = _merge_pair_all_words_with_pair_deltas(chunk, left, right, merged)
                    _dump_words_chunk(chunk_path, chunk)

                for pair, d in delta.items():
                    total_delta[pair] = total_delta.get(pair, 0) + d

            if total_delta:
                total_delta = {p: v for p, v in total_delta.items() if v != 0}

            elapsed_s = time.perf_counter() - t0
            result_q.put(("merge_delta", total_delta, elapsed_s))

        elif cmd == "stop":
            break


class _BPEStreamWorkerPool:
    """流式模式的 worker pool，每个 worker 负责处理一部分 chunk 文件。"""
    def __init__(self, chunk_files: list[Path], num_workers: int, use_inverted_index: bool) -> None:
        n = len(chunk_files)
        chunk_size = max(1, (n + num_workers - 1) // num_workers)
        self._workers: list[tuple[_mp.Process, _mp.Queue, _mp.Queue]] = []  # type: ignore[type-arg]

        for i in range(num_workers):
            start = i * chunk_size
            if start >= n:
                break
            end = min(start + chunk_size, n)
            worker_chunk_files = chunk_files[start:end]

            cmd_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
            result_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
            p = _mp.Process(
                target=_stream_worker_main,
                args=(worker_chunk_files, use_inverted_index, cmd_q, result_q),
                daemon=True,
            )
            p.start()
            self._workers.append((p, cmd_q, result_q))

    def _worker_count(self) -> int:
        return len(self._workers)

    @property
    def alive(self) -> bool:
        return all(p.is_alive() for p, _, _ in self._workers)

    def count_pairs(self) -> tuple[Counter[tuple[bytes, bytes]], dict[str, Any]]:
        """统计所有 chunk 文件中的字节对频率。"""
        worker_n = self._worker_count()
        t_enq0 = time.perf_counter()
        for _, cmd_q, _ in self._workers:
            cmd_q.put("count")
        t_enq1 = time.perf_counter()

        total: Counter[tuple[bytes, bytes]] = Counter()
        t_deq0 = time.perf_counter()
        consumer_task_times_s: list[float] = []
        for _, _, result_q in self._workers:
            _kind, partial, elapsed_s = result_q.get()
            consumer_task_times_s.append(float(elapsed_s))
            for pair, freq in partial.items():
                total[pair] += freq
        t_deq1 = time.perf_counter()

        enqueue_dur_s = max(1e-12, t_enq1 - t_enq0)
        dequeue_dur_s = max(1e-12, t_deq1 - t_deq0)
        consumer_avg_task_ms = 1000.0 * (sum(consumer_task_times_s) / max(1, len(consumer_task_times_s)))

        metrics = {
            "tasks_enqueued": worker_n,
            "tasks_dequeued": worker_n,
            "enqueue_ms": 1000.0 * (t_enq1 - t_enq0),
            "dequeue_wait_ms": 1000.0 * (t_deq1 - t_deq0),
            "enqueue_throughput_tps": worker_n / enqueue_dur_s,
            "dequeue_throughput_tps": worker_n / dequeue_dur_s,
            "consumer_avg_task_ms": consumer_avg_task_ms,
        }
        return total, metrics

    def merge_pair_with_pair_deltas(
        self, left: bytes, right: bytes, merged: bytes
    ) -> tuple[dict[tuple[bytes, bytes], int], dict[str, Any]]:
        """在所有 chunk 文件中执行 merge 操作，并返回频率增量。"""
        worker_n = self._worker_count()
        t_enq0 = time.perf_counter()
        cmd = ("merge_delta", left, right, merged)
        for _, cmd_q, _ in self._workers:
            cmd_q.put(cmd)
        t_enq1 = time.perf_counter()

        total_delta: dict[tuple[bytes, bytes], int] = {}
        consumer_task_times_s: list[float] = []
        t_deq0 = time.perf_counter()

        for _, _, result_q in self._workers:
            _kind, partial_delta, elapsed_s = result_q.get()
            consumer_task_times_s.append(float(elapsed_s))
            for pair, d in partial_delta.items():
                total_delta[pair] = total_delta.get(pair, 0) + int(d)
        t_deq1 = time.perf_counter()

        if total_delta:
            total_delta = {p: v for p, v in total_delta.items() if v != 0}

        enqueue_dur_s = max(1e-12, t_enq1 - t_enq0)
        dequeue_dur_s = max(1e-12, t_deq1 - t_deq0)
        consumer_avg_task_ms = 1000.0 * (sum(consumer_task_times_s) / max(1, len(consumer_task_times_s)))

        metrics = {
            "tasks_enqueued": worker_n,
            "tasks_dequeued": worker_n,
            "enqueue_ms": 1000.0 * (t_enq1 - t_enq0),
            "dequeue_wait_ms": 1000.0 * (t_deq1 - t_deq0),
            "enqueue_throughput_tps": worker_n / enqueue_dur_s,
            "dequeue_throughput_tps": worker_n / dequeue_dur_s,
            "consumer_avg_task_ms": consumer_avg_task_ms,
        }
        return total_delta, metrics

    def shutdown(self) -> None:
        for _, cmd_q, _ in self._workers:
            try:
                cmd_q.put("stop")
            except Exception:
                pass
        for p, _, _ in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


def _save_checkpoint(path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab": {str(k): list(v) for k, v in vocab.items()},
        "merges": [[list(a), list(b)] for a, b in merges],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_checkpoint(path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    vocab = {int(k): bytes(v) for k, v in payload["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in payload["merges"]]
    return vocab, merges


_PARALLEL_WORD_THRESHOLD = 5000


def _dump_words_chunk(path: Path, words: list[list[bytes]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_words_chunk(path: Path) -> list[list[bytes]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _dump_words_chunk_with_index(path: Path, chunk: Any) -> None:
    chunk.save(path)


def _load_words_chunk_with_index(path: Path) -> Any:
    from ._rust_bridge import WordsChunkWithIndex

    return WordsChunkWithIndex.load(path)


def _get_system_memory_percent() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        if total > 0:
            return 100.0 * (1.0 - available / total)
    except Exception:
        pass

    return None


def _force_release_memory() -> None:
    import gc

    gc.collect()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _load_chunk_batch(
    chunk_files: list[Path],
    start_idx: int,
    memory_target_percent: float,
    use_inverted_index: bool = False,
) -> tuple[list[tuple[Path, Any]], int]:
    loaded: list[tuple[Path, Any]] = []
    idx = start_idx

    conservative_threshold = max(50.0, memory_target_percent - 25.0)

    while idx < len(chunk_files):
        if loaded:
            mem = _get_system_memory_percent()
            if mem is not None and mem >= conservative_threshold:
                break
        path = chunk_files[idx]

        if use_inverted_index:
            chunk = _load_words_chunk_with_index(path)
        else:
            chunk = _load_words_chunk(path)

        loaded.append((path, chunk))
        idx += 1

        if loaded:
            mem = _get_system_memory_percent()
            if mem is not None and mem >= conservative_threshold:
                break

    return loaded, idx


def _init_rust_acceleration() -> None:
    global RUST_AVAILABLE, _rust_count_pairs_fn, _rust_merge_pair_all_words_with_deltas_fn
    try:
        from ._rust_bridge import (
            RUST_AVAILABLE as _ok,
            count_pairs as _cp,
            merge_pair_all_words_with_deltas as _mpd,
        )

        RUST_AVAILABLE = bool(_ok)
        _rust_count_pairs_fn = _cp
        _rust_merge_pair_all_words_with_deltas_fn = _mpd
        if RUST_AVAILABLE:
            _log.info("Rust acceleration enabled for BPE training")
    except ImportError:
        RUST_AVAILABLE = False
        _rust_count_pairs_fn = None
        _rust_merge_pair_all_words_with_deltas_fn = None
        _log.info("Rust acceleration not available, using Python implementation")


_init_rust_acceleration()
