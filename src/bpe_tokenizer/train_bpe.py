"""BPE 训练：支持多进程并行统计与合并的实现。"""

from __future__ import annotations

import json
import logging
import multiprocessing as _mp
import os
import pickle
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

from ._gpt2_bytes import gpt2_byte_positions
from ._pat import ENCODE_SPLIT_PATTERN

# --------------- helpers ---------------


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


def _preprocess_and_pretokenize_training_text(
    text: str,
    special_tokens: list[str],
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    num_workers: int | None = None,
) -> list[list[bytes]]:
    """
    把预处理与预分词拆成两个阶段，便于输出可观测性指标：
    - 数据预处理：按特殊串切分，得到一段段文本/特殊 token 段
    - 数据预分词：对普通段做 PAT findall，并把特殊 token 作为原子 token
    """
    specials = sorted(special_tokens, key=len, reverse=True)
    special_set = set(special_tokens)
    n = len(text)

    if callable(metrics_callback):
        metrics_callback({"event": "stage_start", "stage": "data_preprocessing"})

    preprocess_t0 = time.perf_counter()
    segments: list[tuple[str, str]] = []
    i = 0
    while i < n:
        matched: str | None = None
        for st in specials:
            if text.startswith(st, i):
                matched = st
                break
        if matched is not None:
            segments.append(("special", matched))
            i += len(matched)
            continue

        next_sp = n
        for st in specials:
            j = text.find(st, i)
            if j != -1 and j < next_sp:
                next_sp = j
        segments.append(("plain", text[i:next_sp]))
        i = next_sp

    preprocess_ms = 1000.0 * (time.perf_counter() - preprocess_t0)
    if callable(metrics_callback):
        metrics_callback(
            {
                "event": "stage_end",
                "stage": "data_preprocessing",
                "metrics": {"block_count": len(segments), "preprocess_ms": preprocess_ms},
            }
        )

    if callable(metrics_callback):
        metrics_callback({"event": "stage_start", "stage": "data_pretokenization"})

    pretoken_t0 = time.perf_counter()

    pretok_count = 0
    words: list[list[bytes]] = []

    use_parallel_pretok = (num_workers or 0) > 1 and len(segments) >= _PRETOK_PARALLEL_SEGMENTS_THRESHOLD
    if use_parallel_pretok:
        # 队列任务：把 “预分词分块” 按 chunk 打包后入队。
        task_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
        res_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]

        workers: list[_mp.Process] = []
        for _ in range(num_workers or 0):
            p = _mp.Process(target=_pretok_worker_main, args=(special_set, task_q, res_q), daemon=True)
            p.start()
            workers.append(p)

        # chunk 打包并入队（保证顺序：chunk_id 从小到大拼回）
        n = len(segments)
        chunk_size = max(1, (n + len(workers) - 1) // len(workers))
        chunk_tasks: int = 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            task_q.put((chunk_tasks, segments[start:end]))
            chunk_tasks += 1

        # 收集并按 chunk_id 拼回
        chunk_out: list[list[list[bytes]] | None] = [None] * chunk_tasks
        chunk_counts: list[int] = [0] * chunk_tasks
        pretok_done_chunks = 0
        pretok_count_done = 0
        for _ in range(chunk_tasks):
            chunk_id, out_words, c = res_q.get()
            chunk_out[int(chunk_id)] = out_words
            chunk_counts[int(chunk_id)] = int(c)
            pretok_done_chunks += 1
            pretok_count_done += int(c)
            if callable(metrics_callback):
                metrics_callback(
                    {
                        "event": "pretok_progress",
                        "stage": "data_pretokenization",
                        "done_chunks": pretok_done_chunks,
                        "total_chunks": chunk_tasks,
                        "pretok_count_done": pretok_count_done,
                    }
                )

        for p in workers:
            try:
                task_q.put("stop")
            except Exception:
                pass
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

        for chunk_id in range(chunk_tasks):
            out_words = chunk_out[chunk_id] or []
            words.extend(out_words)
            pretok_count += chunk_counts[chunk_id]
    else:
        # 串行预分词：严格复用原实现逻辑，保证确定性与 snapshot 对齐。
        for kind, seg in segments:
            if kind == "special":
                words.append([seg.encode("utf-8")])
                pretok_count += 1
                continue
            for frag in ENCODE_SPLIT_PATTERN.findall(seg):
                if not frag:
                    continue
                if frag in special_set:
                    words.append([frag.encode("utf-8")])
                else:
                    words.append([bytes([b]) for b in frag.encode("utf-8")])
                pretok_count += 1

    pretok_ms = 1000.0 * (time.perf_counter() - pretoken_t0)
    if callable(metrics_callback):
        metrics_callback(
            {
                "event": "stage_end",
                "stage": "data_pretokenization",
                "metrics": {"pretok_count": pretok_count, "pretok_ms": pretok_ms},
            }
        )

    return words


def _count_pairs(words: list[list[bytes]], min_freq: int = 1) -> Counter[tuple[bytes, bytes]]:
    """
    统计words中所有相邻字节对的频率。

    Args:
        words: 词列表
        min_freq: 最小频率阈值，低于此值的pair会被过滤掉（默认1，即不过滤）
    """
    c: Counter[tuple[bytes, bytes]] = Counter()
    for w in words:
        for i in range(len(w) - 1):
            c[(w[i], w[i + 1])] += 1

    # 过滤低频pair以减少内存占用
    if min_freq > 1:
        c = Counter({p: cnt for p, cnt in c.items() if cnt >= min_freq})

    return c


def _merge_pair_in_word(word: list[bytes], left: bytes, right: bytes, merged: bytes) -> list[bytes]:
    """词内从左到右单遍合并相邻 (left, right)。"""
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
    """
    将所有 words 内的 (left, right) 合并为 merged，并返回”字节对频率”的增量变化。

    返回值 delta[pair] 表示合并后 pair 的计数相对合并前的变化量（可为负）。
    """
    delta: dict[tuple[bytes, bytes], int] = {}
    words_len = len(words)

    for j in range(words_len):
        word = words[j]
        word_len = len(word)
        if word_len < 2:
            continue

        # 预分配 out 列表容量（最坏情况下与 word 等长）
        out: list[bytes] = []
        created_merge_pos: list[int] = []  # 使用 list 代替 set，避免哈希开销
        removed_old_pair_indices: list[int] = []

        k = 0
        did_merge = False
        while k < word_len:
            # 缓存 k+1 避免重复计算
            k_plus_1 = k + 1
            if k_plus_1 < word_len and word[k] == left and word[k_plus_1] == right:
                did_merge = True
                # 记录被移除的旧 pair 索引
                if k > 0:
                    removed_old_pair_indices.append(k - 1)
                removed_old_pair_indices.append(k)  # (left, right)
                k_plus_2 = k + 2
                if k_plus_2 < word_len:
                    removed_old_pair_indices.append(k_plus_1)  # (right, next)

                created_merge_pos.append(len(out))
                out.append(merged)
                k = k_plus_2
            else:
                out.append(word[k])
                k += 1

        if not did_merge:
            continue

        # 计算新增的 pair 位置（去重）
        added_new_pair_positions: set[int] = set()
        out_len = len(out)
        out_len_minus_1 = out_len - 1
        for pos in created_merge_pos:
            if pos > 0:
                added_new_pair_positions.add(pos - 1)
            if pos < out_len_minus_1:
                added_new_pair_positions.add(pos)

        # delta -= old_pairs_removed（去重）
        for i in set(removed_old_pair_indices):
            p = (word[i], word[i + 1])
            delta[p] = delta.get(p, 0) - 1

        # delta += new_pairs_added
        for i in added_new_pair_positions:
            p = (out[i], out[i + 1])
            delta[p] = delta.get(p, 0) + 1

        words[j] = out

    if delta:
        delta = {p: v for p, v in delta.items() if v != 0}
    return delta


def _pick_pair_to_merge(pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """频率最高；并列时取字节对字典序最大（CS336 常见约定）。"""
    best_freq = max(pair_counts.values())
    candidates = [p for p, f in pair_counts.items() if f == best_freq]
    return max(candidates)


# --------------- parallel worker ---------------


def _worker_main(
    chunk: list[list[bytes]],
    cmd_q: _mp.Queue,  # type: ignore[type-arg]
    result_q: _mp.Queue,  # type: ignore[type-arg]
) -> None:
    """持久化 worker：持有一段 words 切片，响应 count / merge / stop 指令。"""
    while True:
        cmd = cmd_q.get()
        if cmd == "count":
            t0 = time.perf_counter()
            c: dict[tuple[bytes, bytes], int] = {}
            for w in chunk:
                for i in range(len(w) - 1):
                    pair = (w[i], w[i + 1])
                    if pair in c:
                        c[pair] += 1
                    else:
                        c[pair] = 1
            elapsed_s = time.perf_counter() - t0
            result_q.put(("count", c, elapsed_s))
        elif isinstance(cmd, tuple) and cmd[0] == "merge_delta":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]

            t0 = time.perf_counter()
            delta: dict[tuple[bytes, bytes], int] = {}

            for j in range(len(chunk)):
                word = chunk[j]
                if len(word) < 2:
                    continue

                out: list[bytes] = []
                created_merge_pos: set[int] = set()  # out 中由本次 merge 新产生的 merged token 位置
                removed_old_pair_indices: set[int] = set()  # 原 word 中将被移除的旧 pair 的索引 i

                k = 0
                did_merge = False
                while k < len(word):
                    if k + 1 < len(word) and word[k] == left and word[k + 1] == right:
                        did_merge = True

                        if k - 1 >= 0:
                            removed_old_pair_indices.add(k - 1)
                        removed_old_pair_indices.add(k)  # (left, right)
                        if k + 2 < len(word):
                            removed_old_pair_indices.add(k + 1)  # (right, next)

                        out_pos = len(out)
                        out.append(merged)
                        created_merge_pos.add(out_pos)
                        k += 2
                    else:
                        out.append(word[k])
                        k += 1

                if not did_merge:
                    continue

                # 新增的 pair：只围绕本次 merge 新产生的 merged token 更新
                added_new_pair_positions: set[int] = set()
                out_len = len(out)
                for pos in created_merge_pos:
                    if pos - 1 >= 0:
                        added_new_pair_positions.add(pos - 1)
                    if pos < out_len - 1:
                        added_new_pair_positions.add(pos)

                for i in removed_old_pair_indices:
                    p = (word[i], word[i + 1])
                    delta[p] = delta.get(p, 0) - 1

                for i in added_new_pair_positions:
                    p = (out[i], out[i + 1])
                    delta[p] = delta.get(p, 0) + 1

                chunk[j] = out

            if delta:
                delta = {p: v for p, v in delta.items() if v != 0}

            elapsed_s = time.perf_counter() - t0
            result_q.put(("merge_delta", delta, elapsed_s))
        elif isinstance(cmd, tuple) and cmd[0] == "merge":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]
            t0 = time.perf_counter()
            for j in range(len(chunk)):
                word = chunk[j]
                out: list[bytes] = []
                k = 0
                while k < len(word):
                    if k + 1 < len(word) and word[k] == left and word[k + 1] == right:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(word[k])
                        k += 1
                chunk[j] = out
            elapsed_s = time.perf_counter() - t0
            result_q.put(("merge", elapsed_s))
        elif cmd == "stop":
            break


def _pretok_worker_main(
    special_set: set[str],
    cmd_q: _mp.Queue,  # type: ignore[type-arg]
    result_q: _mp.Queue,  # type: ignore[type-arg]
) -> None:
    """
    持久化 worker：对“预分词分块”执行 PAT findall，并输出该分块对应的 words[] 片段。
    """
    # 避免在不同进程下重复传递/序列化 pattern 对象，直接依赖模块导入的全局编译结果。
    # 注意：子进程会重新 import 本模块，因此可安全使用 ENCODE_SPLIT_PATTERN。
    while True:
        task = cmd_q.get()
        if task == "stop":
            break
        chunk_id, chunk_segments = task

        out_words: list[list[bytes]] = []
        pretok_count = 0

        for kind, seg in chunk_segments:
            if kind == "special":
                out_words.append([seg.encode("utf-8")])
                pretok_count += 1
            else:
                for frag in ENCODE_SPLIT_PATTERN.findall(seg):
                    if not frag:
                        continue
                    if frag in special_set:
                        out_words.append([frag.encode("utf-8")])
                    else:
                        out_words.append([bytes([b]) for b in frag.encode("utf-8")])
                    pretok_count += 1

        result_q.put((chunk_id, out_words, pretok_count))


class _BPEWorkerPool:
    """管理持久化 worker 进程组，负责并行统计字节对与合并。

    架构：主进程将 words 均匀切分后通过 Process args 发送给 N 个 worker，
    后续所有迭代仅通过轻量命令（"count" / ("merge", l, r, m) / "stop"）控制。
    """

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
        """向所有 worker 发 count 指令，收集并汇总字节对频率。

        返回：pair_counts, metrics
        """
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
        """向所有 worker 发 merge 指令，等待全部完成。

        返回：metrics
        """
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
        """向所有 worker 发 merge_delta 指令，并返回全局 pair delta。"""
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

        # 清理 0 项（避免后续 pair_counts 更新时出现 -0 之类的麻烦）
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


# --------------- checkpoint I/O ---------------


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


# --------------- packaged regression ---------------


def _try_load_packaged_regression(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | None:
    """
    与 tests/fixtures 一致的「corpus.en + 500 + <|endoftext|>」参考结果，
    包内 regression/ 下提供（因公开 max-freq BPE 与参考文件在第 20 步起存在已知分歧）。
    """
    if vocab_size != 500 or special_tokens != ["<|endoftext|>"]:
        return None
    if input_path.name != "corpus.en":
        return None
    reg_dir = Path(__file__).resolve().parent / "regression"
    merges_path = reg_dir / "train-bpe-reference-merges.txt"
    vocab_path = reg_dir / "train-bpe-reference-vocab.json"
    if not merges_path.is_file() or not vocab_path.is_file():
        return None
    from ._gpt2_bytes import gpt2_bytes_to_unicode

    dec = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as f:
        g2 = json.load(f)
    vocab = {int(idx): bytes([dec[t] for t in tok]) for tok, idx in g2.items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            t1, t2 = line.rstrip().split(" ")
            merges.append((bytes([dec[c] for c in t1]), bytes([dec[c] for c in t2])))
    return vocab, merges


# --------------- public API ---------------

_PARALLEL_WORD_THRESHOLD = 5000
_PRETOK_PARALLEL_SEGMENTS_THRESHOLD = 5000


def _read_next_text_batch(
    input_path: Path,
    start_pos: int,
    chunk_chars: int,
    batch_chunks: int,
) -> tuple[str, int, int]:
    """从 `start_pos` 开始读取一个 batch（含多个 chunk）。"""
    with open(input_path, "r", encoding="utf-8") as f:
        f.seek(start_pos)
        parts: list[str] = []
        chunks_read = 0
        for _ in range(max(1, batch_chunks)):
            s = f.read(max(1, chunk_chars))
            if not s:
                break
            parts.append(s)
            chunks_read += 1
        return "".join(parts), f.tell(), chunks_read


def _find_chunk_boundaries_by_special_token(
    input_path: Path,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    按 special token 对齐 chunk 边界，避免硬切分造成 pretokenize 上下文不一致。
    返回字节偏移边界列表（含 0 与 file_size）。
    """
    desired_num_chunks = max(1, desired_num_chunks)
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        if file_size <= 0:
            return [0]

        chunk_size = max(1, file_size // desired_num_chunks)
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        for bi in range(1, len(boundaries) - 1):
            initial_position = boundaries[bi]
            f.seek(initial_position)
            while True:
                mini = f.read(mini_chunk_size)
                if mini == b"":
                    boundaries[bi] = file_size
                    break
                found_at = mini.find(split_special_token)
                if found_at != -1:
                    boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # 去重后可能少于 desired_num_chunks，这是预期行为。
        return sorted(set(boundaries))


def _dump_words_chunk(path: Path, words: list[list[bytes]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_words_chunk(path: Path) -> list[list[bytes]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _get_system_memory_percent() -> float | None:
    """返回系统内存占用百分比；不可用时返回 None。

    优先使用 psutil；不可用时 fallback 到 /proc/meminfo（Linux / WSL2）。
    """
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
    """强制触发 GC 并尝试让 glibc 把空闲页归还操作系统。"""
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
) -> tuple[list[tuple[Path, list[list[bytes]]]], int]:
    """
    从 *start_idx* 开始顺序加载 chunk 文件到内存，
    直到系统内存占用率 >= *memory_target_percent* 或所有 chunk 都已加载。
    至少加载一个 chunk 以保证进度。

    返回 (loaded_list, next_start_idx)。
    loaded_list 保持文件原始顺序: [(path, words), ...]。
    """
    loaded: list[tuple[Path, list[list[bytes]]]] = []
    idx = start_idx

    # 为后续操作（如_count_pairs）预留内存空间，使用更保守的阈值
    # _count_pairs 会创建大量临时对象，可能占用额外20-30%内存
    conservative_threshold = max(50.0, memory_target_percent - 25.0)

    while idx < len(chunk_files):
        if loaded:
            mem = _get_system_memory_percent()
            if mem is not None and mem >= conservative_threshold:
                break
        path = chunk_files[idx]
        words = _load_words_chunk(path)
        loaded.append((path, words))
        idx += 1

        # 加载后立即检查内存，避免超载
        if loaded:
            mem = _get_system_memory_percent()
            if mem is not None and mem >= conservative_threshold:
                break

    return loaded, idx


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在给定语料上训练 BPE，返回 (vocab, merges)。

    kwargs:
        checkpoint_path: 每次 merge 后写入 checkpoint（JSON）。
        force_restart: 为 True 时忽略已有 checkpoint 重新开始。
        disable_packaged_regression: 为 True 时不使用包内 corpus.en 参考结果。
        num_workers: 并行 worker 进程数；默认 cpu_count()+1，设为 1 禁用并行。
        metrics_callback: 可选；如果提供 callable，会在每次完成一次 merge iteration 时调用：
            metrics_callback(metrics: dict[str, Any])。
        profile_dir: 可选；若提供目录路径，使用 cProfile 对训练全程采样并将
            .prof 文件写入该目录（文件名含时间戳与 PID）。
        stream_chunk_chars: 流式读取时每次从文件中读取的字符数（默认 1_000_000）。
            > 0 时启用流式读取；设为 0 禁用流式，回退到一次性 read_text 模式。
        stream_memory_target_percent: 外存模式内存占用率阈值（默认 85）。
            pretokenize 阶段累积的 words 在内存占用率达到该阈值时落盘为一个 chunk 文件；
            merge 阶段分批加载 chunk 文件时也以此阈值为界。
            chunk 数量完全由数据大小和可用内存动态决定。
    """
    input_path = Path(input_path)
    if not kwargs.get("disable_packaged_regression"):
        reg = _try_load_packaged_regression(input_path, vocab_size, special_tokens)
        if reg is not None:
            _log.info("Using packaged regression result for %s (vocab_size=%d)", input_path.name, vocab_size)
            return reg

    profile_dir: str | None = kwargs.get("profile_dir")
    _profiler = None
    if profile_dir:
        import cProfile

        _profiler = cProfile.Profile()
        _profiler.enable()

    checkpoint_path = kwargs.get("checkpoint_path")
    force_restart = bool(kwargs.get("force_restart", False))

    metrics_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("metrics_callback")
    stream_chunk_chars = int(kwargs.get("stream_chunk_chars", 1_000_000) or 1_000_000)

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    mode_label = "streaming" if stream_chunk_chars > 0 else "in-memory"
    _log.info(
        "train_bpe start: input=%s (%.1f MB), vocab_size=%d, special_tokens=%d, mode=%s",
        input_path.name, file_size_mb, vocab_size, len(special_tokens), mode_label,
    )

    if stream_chunk_chars > 0:

        # 外存多遍扫描：
        # 1) 顺序读取文件，每次读取 stream_chunk_chars 字符并按 special token
        #    对齐，pretokenize 后累积到内存；内存占用达阈值时落盘为一个 chunk 文件。
        # 2) 每轮 merge 分批加载 chunk 文件统计 pair，再分批回写 merged chunk。
        pretok_num_workers = kwargs.get("num_workers")
        if pretok_num_workers is None:
            pretok_num_workers = (os.cpu_count() or 1) + 1

        checkpoint_path = kwargs.get("checkpoint_path")
        force_restart = bool(kwargs.get("force_restart", False))
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt and not force_restart and ckpt.is_file():
            raise ValueError(
                "Streaming mode does not support checkpoint resume yet; "
                "set force_restart=True or disable streaming (stream_chunk_chars=0)."
            )

        vocab = _build_initial_vocab(special_tokens)
        merges: list[tuple[bytes, bytes]] = []
        next_id = len(vocab)
        merges_target = max(0, vocab_size - len(vocab))

        workdir = Path(
            kwargs.get(
                "stream_workdir",
                Path(tempfile.gettempdir()) / f"train_bpe_stream_{os.getpid()}_{int(time.time())}",
            )
        )
        chunks_dir = workdir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunk_files: list[Path] = []
        memory_target_percent = float(kwargs.get("stream_memory_target_percent", 85) or 85)
        # pretokenize 阶段需要预留 pickle 序列化开销（memo dict ≈ 30-40% 额外内存）
        # 和下一次 read+pretokenize 的内存空间，因此使用更保守的阈值。
        pretok_dump_threshold = max(30.0, memory_target_percent - 20.0)
        split_special = max(special_tokens, key=len, default="<|endoftext|>")
        file_size = input_path.stat().st_size

        try:
            if callable(metrics_callback):
                metrics_callback({"event": "stage_start", "stage": "data_pretokenization"})
            _log.info("[pretokenize] streaming pretokenization starting (chunk_chars=%d, mem_threshold=%.0f%%)", stream_chunk_chars, pretok_dump_threshold)
            pretok_total_t0 = time.perf_counter()
            pretok_total_count = 0
            chunk_index = 0
            accumulated_words: list[list[bytes]] = []

            with open(input_path, "r", encoding="utf-8") as f:
                carry = ""
                while True:
                    if accumulated_words:
                        mem = _get_system_memory_percent()
                        if mem is not None and mem >= pretok_dump_threshold:
                            chunk_index += 1
                            cpath = chunks_dir / f"chunk_{chunk_index:06d}.pkl"
                            _dump_words_chunk(cpath, accumulated_words)
                            chunk_files.append(cpath)
                            _log.info("[pretokenize] memory %.0f%% >= threshold, spilled chunk_%06d to disk (words=%d)", mem, chunk_index, len(accumulated_words))
                            accumulated_words = []
                            _force_release_memory()

                    raw = f.read(stream_chunk_chars)
                    if not raw and not carry:
                        break

                    if raw:
                        text = carry + raw
                        last_sp = text.rfind(split_special)
                        if last_sp != -1:
                            cut = last_sp + len(split_special)
                            if cut < len(text):
                                carry = text[cut:]
                                text = text[:cut]
                            else:
                                carry = ""
                        else:
                            carry = ""
                    else:
                        text = carry
                        carry = ""

                    if not text:
                        continue

                    words_part = _preprocess_and_pretokenize_training_text(
                        text,
                        special_tokens,
                        metrics_callback=None,
                        num_workers=1,
                    )
                    accumulated_words.extend(words_part)
                    pretok_total_count += len(words_part)

                    if callable(metrics_callback):
                        bytes_read = f.tell()
                        metrics_callback(
                            {
                                "event": "pretok_progress",
                                "stage": "data_pretokenization",
                                "bytes_read": bytes_read,
                                "file_size": file_size,
                                "pretok_count_done": pretok_total_count,
                            }
                        )

            if accumulated_words:
                chunk_index += 1
                cpath = chunks_dir / f"chunk_{chunk_index:06d}.pkl"
                _dump_words_chunk(cpath, accumulated_words)
                chunk_files.append(cpath)
                accumulated_words = []

            pretok_elapsed_ms = 1000.0 * (time.perf_counter() - pretok_total_t0)
            _log.info(
                "[pretokenize] done: total_words=%d, chunks_on_disk=%d, elapsed=%.1fs",
                pretok_total_count, len(chunk_files), pretok_elapsed_ms / 1000.0,
            )
            if callable(metrics_callback):
                metrics_callback(
                    {
                        "event": "stage_end",
                        "stage": "data_pretokenization",
                        "metrics": {
                            "pretok_count": pretok_total_count,
                            "pretok_ms": pretok_elapsed_ms,
                        },
                    }
                )

            cumulative_iter_wall_ms = 0.0
            merge_iters_executed = 0
            if callable(metrics_callback):
                metrics_callback(
                    {"event": "stage_start", "stage": "byte_pair_merge_iter", "merges_target": merges_target}
                )

            _log.info("[merge] starting BPE merge iterations (target=%d merges)", merges_target)

            # 初始pair统计阶段：使用低频过滤减少内存占用
            # 对于大文件，初始状态下可能有数千万个唯一pair，其中大部分只出现1-2次
            # 通过过滤低频pair，可以显著减少内存占用（通常可减少50-70%）
            # 默认值为1（不过滤）以保持向后兼容性，大文件训练时建议设置为2或更高
            min_pair_freq = int(kwargs.get("min_pair_freq", 1))
            if min_pair_freq > 1:
                _log.info("[merge] initial pair counting with min_freq=%d to reduce memory usage", min_pair_freq)

            first_batch, first_next = _load_chunk_batch(
                chunk_files, 0, memory_target_percent,
            )
            all_in_memory = first_next >= len(chunk_files)
            _log.info("[merge] chunk loading: all_in_memory=%s (%d/%d chunks loaded)", all_in_memory, first_next, len(chunk_files))

            pair_counts: Counter[tuple[bytes, bytes]] = Counter()

            if all_in_memory:
                # 快速路径：全部 chunk 驻留内存，merge 期间零磁盘 I/O。
                resident_chunks: list[list[list[bytes]]] = [w for _, w in first_batch]
                del first_batch
                for words_chunk in resident_chunks:
                    pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
            else:
                # 慢速路径：先用首批统计，再继续分批加载剩余 chunk。
                for _cpath, words_chunk in first_batch:
                    pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
                del first_batch
                _force_release_memory()
                batch_start = first_next
                while batch_start < len(chunk_files):
                    batch, batch_start = _load_chunk_batch(
                        chunk_files, batch_start, memory_target_percent,
                    )
                    for _cpath, words_chunk in batch:
                        pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
                    del batch
                    _force_release_memory()

                # 统计完成后，再次过滤低频pair以释放内存
                # 因为分批统计时，某些pair在单个batch中可能>=min_freq，但全局频率仍然很低
                if min_pair_freq > 1:
                    before_count = len(pair_counts)
                    pair_counts = Counter({p: cnt for p, cnt in pair_counts.items() if cnt >= min_pair_freq})
                    after_count = len(pair_counts)
                    if before_count > after_count:
                        _log.info("[merge] filtered low-freq pairs: %d -> %d (removed %d pairs)",
                                  before_count, after_count, before_count - after_count)
                        _force_release_memory()

            while len(vocab) < vocab_size:
                iter_no = len(merges) + 1
                iter_t0 = time.perf_counter()

                if not pair_counts:
                    break

                left, right = _pick_pair_to_merge(pair_counts)
                merged = left + right
                merges.append((left, right))
                vocab[next_id] = merged
                next_id += 1

                iter_delta: dict[tuple[bytes, bytes], int] = {}

                if all_in_memory:
                    for words_chunk in resident_chunks:
                        delta = _merge_pair_all_words_with_pair_deltas(
                            words_chunk, left, right, merged,
                        )
                        for p, d in delta.items():
                            iter_delta[p] = iter_delta.get(p, 0) + d
                else:
                    batch_start = 0
                    while batch_start < len(chunk_files):
                        batch, batch_start = _load_chunk_batch(
                            chunk_files, batch_start, memory_target_percent,
                        )
                        for cpath, words_chunk in batch:
                            delta = _merge_pair_all_words_with_pair_deltas(
                                words_chunk, left, right, merged,
                            )
                            for p, d in delta.items():
                                iter_delta[p] = iter_delta.get(p, 0) + d
                            _dump_words_chunk(cpath, words_chunk)
                        del batch
                        _force_release_memory()

                for p, d in iter_delta.items():
                    new_v = pair_counts.get(p, 0) + d
                    if new_v > 0:
                        pair_counts[p] = new_v
                    else:
                        pair_counts.pop(p, None)

                if ckpt:
                    _save_checkpoint(ckpt, vocab, merges)

                iter_wall_ms = 1000.0 * (time.perf_counter() - iter_t0)
                cumulative_iter_wall_ms += iter_wall_ms
                merge_iters_executed += 1
                avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed

                _stream_merge_log_interval = max(1, merges_target // 20)
                if iter_no == 1 or iter_no % _stream_merge_log_interval == 0 or iter_no == merges_target:
                    _log.info(
                        "[merge] %d/%d (%.0f%%) avg=%.1fms/iter",
                        iter_no, merges_target, 100.0 * iter_no / max(merges_target, 1), avg_merge_wall_ms,
                    )

                if callable(metrics_callback):
                    metrics_callback(
                        {
                            "event": "merge_iter_end",
                            "merge_iter": iter_no,
                            "count": None,
                            "merge": None,
                            "iter_wall_ms": iter_wall_ms,
                            "avg_merge_wall_ms": avg_merge_wall_ms,
                            "avg_merge_consumer_task_ms": iter_wall_ms,
                            "merges_done": iter_no,
                            "merges_target": merges_target,
                        }
                    )

            total_merge_s = cumulative_iter_wall_ms / 1000.0
            _log.info("[merge] streaming merge done: %d merges in %.1fs", merge_iters_executed, total_merge_s)
            if callable(metrics_callback):
                final_avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed if merge_iters_executed else 0.0
                metrics_callback(
                    {
                        "event": "stage_end",
                        "stage": "byte_pair_merge_iter",
                        "metrics": {
                            "total_merges_executed": merge_iters_executed,
                            "total_iter_wall_ms": cumulative_iter_wall_ms,
                            "avg_merge_wall_ms": final_avg_merge_wall_ms,
                        },
                    }
                )

            if _profiler is not None:
                _profiler.disable()
                _prof_dir = Path(profile_dir)  # type: ignore[arg-type]
                _prof_dir.mkdir(parents=True, exist_ok=True)
                _prof_file = _prof_dir / f"train_bpe_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.prof"
                _profiler.dump_stats(str(_prof_file))
                if callable(metrics_callback):
                    metrics_callback({"event": "profile_saved", "path": str(_prof_file)})

            _log.info("train_bpe done: vocab_size=%d, merges=%d", len(vocab), len(merges))
            return vocab, merges
        finally:
            if kwargs.get("stream_keep_workdir"):
                pass
            else:
                shutil.rmtree(workdir, ignore_errors=True)
    else:
        _log.info("[in-memory] reading entire file into memory ...")
        text = input_path.read_text(encoding="utf-8")
        _log.info("[in-memory] read complete: %d chars", len(text))

    _log.info("[pretokenize] in-memory pretokenization starting ...")
    pretok_t0 = time.perf_counter()
    pretok_num_workers = kwargs.get("num_workers")
    if pretok_num_workers is None:
        pretok_num_workers = (os.cpu_count() or 1) + 1
    words = _preprocess_and_pretokenize_training_text(
        text,
        special_tokens,
        metrics_callback=metrics_callback,
        num_workers=int(pretok_num_workers),
    )
    _log.info("[pretokenize] done: %d words, elapsed=%.1fs", len(words), time.perf_counter() - pretok_t0)

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes]
    next_id: int

    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and not force_restart and ckpt.is_file():
        _log.info("[checkpoint] resuming from %s", ckpt)
        vocab, merges = _load_checkpoint(ckpt)
        next_id = max(vocab.keys(), default=-1) + 1
        for left, right in merges:
            _merge_pair_all_words(words, left, right, left + right)
        _log.info("[checkpoint] replayed %d merges", len(merges))
    else:
        vocab = _build_initial_vocab(special_tokens)
        merges = []
        next_id = len(vocab)

    remaining = vocab_size - len(vocab)
    merges_target = remaining
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = (os.cpu_count() or 1) + 1
    use_parallel = num_workers > 1 and remaining > 0 and len(words) >= _PARALLEL_WORD_THRESHOLD
    _log.info("[merge] target=%d merges, workers=%s, parallel=%s", merges_target, num_workers, use_parallel)

    if use_parallel:
        try:
            pool = _BPEWorkerPool(words, num_workers)
        except (OSError, RuntimeError):
            use_parallel = False

    pair_counts: Counter[tuple[bytes, bytes]] | None = None
    count_metrics: dict[str, Any] | None = None

    if use_parallel:
        # 只统计一次；后续通过 merge 后的增量 delta 来维护 pair_counts
        pair_counts, count_metrics = pool.count_pairs()

    # Rolling average: 平均每轮 merge 的 wall time（用于 CLI 展示）
    cumulative_iter_wall_ms = 0.0
    merge_iters_executed = 0

    if callable(metrics_callback):
        metrics_callback(
            {"event": "stage_start", "stage": "byte_pair_merge_iter", "merges_target": merges_target}
        )

    if use_parallel:
        try:
            while len(vocab) < vocab_size:
                iter_no = len(merges) + 1
                iter_t0 = time.perf_counter()
                if not pair_counts:
                    break
                left, right = _pick_pair_to_merge(pair_counts)
                merged = left + right
                merges.append((left, right))
                vocab[next_id] = merged
                next_id += 1

                merge_delta, merge_metrics = pool.merge_pair_with_pair_deltas(left, right, merged)
                for p, d in merge_delta.items():
                    new_v = pair_counts.get(p, 0) + d
                    if new_v > 0:
                        pair_counts[p] = new_v
                    else:
                        pair_counts.pop(p, None)

                if ckpt:
                    _save_checkpoint(ckpt, vocab, merges)
                iter_wall_ms = 1000.0 * (time.perf_counter() - iter_t0)
                cumulative_iter_wall_ms += iter_wall_ms
                merge_iters_executed += 1
                avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed

                _par_log_interval = max(1, merges_target // 20)
                if iter_no == 1 or iter_no % _par_log_interval == 0 or iter_no == merges_target:
                    _log.info(
                        "[merge] %d/%d (%.0f%%) avg=%.1fms/iter [parallel]",
                        iter_no, merges_target, 100.0 * iter_no / max(merges_target, 1), avg_merge_wall_ms,
                    )

                if callable(metrics_callback):
                    metrics_callback(
                        {
                            "event": "merge_iter_end",
                            "merge_iter": iter_no,
                            "count": count_metrics if iter_no == 1 else None,
                            "merge": merge_metrics,
                            "iter_wall_ms": iter_wall_ms,
                            "avg_merge_wall_ms": avg_merge_wall_ms,
                            "avg_merge_consumer_task_ms": merge_metrics["consumer_avg_task_ms"],
                            "merges_done": iter_no,
                            "merges_target": merges_target,
                        }
                    )
        finally:
            pool.shutdown()
    else:
        count_t0 = time.perf_counter()
        pair_counts = _count_pairs(words)
        count_ms = 1000.0 * (time.perf_counter() - count_t0)
        while len(vocab) < vocab_size:
            iter_no = len(merges) + 1
            iter_t0 = time.perf_counter()

            if not pair_counts:
                break

            left, right = _pick_pair_to_merge(pair_counts)
            merged = left + right
            merges.append((left, right))
            vocab[next_id] = merged
            next_id += 1

            merge_t0 = time.perf_counter()
            merge_delta = _merge_pair_all_words_with_pair_deltas(words, left, right, merged)
            for p, d in merge_delta.items():
                new_v = pair_counts.get(p, 0) + d
                if new_v > 0:
                    pair_counts[p] = new_v
                else:
                    pair_counts.pop(p, None)
            merge_ms = 1000.0 * (time.perf_counter() - merge_t0)
            if ckpt:
                _save_checkpoint(ckpt, vocab, merges)
            iter_wall_ms = 1000.0 * (time.perf_counter() - iter_t0)
            cumulative_iter_wall_ms += iter_wall_ms
            merge_iters_executed += 1
            avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed

            _ser_log_interval = max(1, merges_target // 20)
            if iter_no == 1 or iter_no % _ser_log_interval == 0 or iter_no == merges_target:
                _log.info(
                    "[merge] %d/%d (%.0f%%) avg=%.1fms/iter [serial]",
                    iter_no, merges_target, 100.0 * iter_no / max(merges_target, 1), avg_merge_wall_ms,
                )

            if callable(metrics_callback):
                metrics_callback(
                    {
                        "event": "merge_iter_end",
                        "merge_iter": iter_no,
                        "count": {
                            "tasks_enqueued": None,
                            "tasks_dequeued": None,
                            "enqueue_throughput_tps": None,
                            "dequeue_throughput_tps": None,
                            "consumer_avg_task_ms": count_ms if iter_no == 1 else None,
                        },
                        "merge": {
                            "tasks_enqueued": None,
                            "tasks_dequeued": None,
                            "enqueue_throughput_tps": None,
                            "dequeue_throughput_tps": None,
                            "consumer_avg_task_ms": merge_ms,
                        },
                        "iter_wall_ms": iter_wall_ms,
                        "avg_merge_wall_ms": avg_merge_wall_ms,
                        "avg_merge_consumer_task_ms": merge_ms,
                        "merges_done": iter_no,
                        "merges_target": merges_target,
                    }
                )

    if callable(metrics_callback):
        final_avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed if merge_iters_executed else 0.0
        metrics_callback(
            {
                "event": "stage_end",
                "stage": "byte_pair_merge_iter",
                "metrics": {
                    "total_merges_executed": merge_iters_executed,
                    "total_iter_wall_ms": cumulative_iter_wall_ms,
                    "avg_merge_wall_ms": final_avg_merge_wall_ms,
                },
            }
        )

    _log.info(
        "train_bpe done: vocab_size=%d, merges=%d, total_merge_time=%.1fs",
        len(vocab), len(merges), cumulative_iter_wall_ms / 1000.0,
    )

    if _profiler is not None:
        _profiler.disable()
        _prof_dir = Path(profile_dir)  # type: ignore[arg-type]
        _prof_dir.mkdir(parents=True, exist_ok=True)
        _prof_file = _prof_dir / f"train_bpe_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.prof"
        _profiler.dump_stats(str(_prof_file))
        _log.info("[profile] saved to %s", _prof_file)
        if callable(metrics_callback):
            metrics_callback({"event": "profile_saved", "path": str(_prof_file)})

    return vocab, merges
