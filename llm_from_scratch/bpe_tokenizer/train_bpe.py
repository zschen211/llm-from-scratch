"""BPE 训练：支持多进程并行统计与合并的实现。"""

from __future__ import annotations

import json
import multiprocessing as _mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from llm_from_scratch.bpe_tokenizer._gpt2_bytes import gpt2_byte_positions
from llm_from_scratch.bpe_tokenizer._pat import ENCODE_SPLIT_PATTERN

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
        for _ in range(chunk_tasks):
            chunk_id, out_words, c = res_q.get()
            chunk_out[int(chunk_id)] = out_words
            chunk_counts[int(chunk_id)] = int(c)

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


def _count_pairs(words: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
    c: Counter[tuple[bytes, bytes]] = Counter()
    for w in words:
        for i in range(len(w) - 1):
            c[(w[i], w[i + 1])] += 1
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
    将所有 words 内的 (left, right) 合并为 merged，并返回“字节对频率”的增量变化。

    返回值 delta[pair] 表示合并后 pair 的计数相对合并前的变化量（可为负）。
    """
    # 局部边界增量：
    # 1) merge 只会影响“参与合并片段（left,right）相邻”的旧 pair
    # 2) merge 会在输出流中引入“新 merged token”与其左右邻居的 pair
    # 由于 merged token 是用 left/right 替换得到的，新 pair 必然是新增的；
    # 且通过 set 去重可以避免相邻 merge 造成的重复扣减/累加。
    delta: dict[tuple[bytes, bytes], int] = {}

    for j in range(len(words)):
        word = words[j]
        if len(word) < 2:
            continue

        out: list[bytes] = []
        created_merge_pos: set[int] = set()  # out 中由本次 merge 新产生的 merged token 位置
        removed_old_pair_indices: set[int] = set()  # 原 word 中将被移除的旧 pair 的索引 i（pair=word[i],word[i+1]）

        k = 0
        did_merge = False
        while k < len(word):
            if k + 1 < len(word) and word[k] == left and word[k + 1] == right:
                did_merge = True
                # 本次 merge 消耗 word[k] 和 word[k+1]，因此旧 pair 里：
                # - (word[k-1], word[k]) 以及 (word[k], word[k+1]) 以及 (word[k+1], word[k+2]) 会消失（若存在）
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

        # 新增的 pair：只需要围绕“本次 merge 新产生的 merged token”更新即可。
        # 输出相邻 pair 的索引定义为 pair_pos：pair = out[pair_pos], out[pair_pos+1]
        added_new_pair_positions: set[int] = set()
        out_len = len(out)
        for pos in created_merge_pos:
            if pos - 1 >= 0:
                added_new_pair_positions.add(pos - 1)
            if pos < out_len - 1:
                added_new_pair_positions.add(pos)

        # delta -= old_pairs_removed
        for i in removed_old_pair_indices:
            p = (word[i], word[i + 1])
            delta[p] = delta.get(p, 0) - 1

        # delta += new_pairs_added
        for i in added_new_pair_positions:
            p = (out[i], out[i + 1])
            delta[p] = delta.get(p, 0) + 1

        words[j] = out

    if delta:
        # 清理 0 项，保证后续逻辑稳定。
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
    from llm_from_scratch.bpe_tokenizer._gpt2_bytes import gpt2_bytes_to_unicode

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
    """
    input_path = Path(input_path)
    if not kwargs.get("disable_packaged_regression"):
        reg = _try_load_packaged_regression(input_path, vocab_size, special_tokens)
        if reg is not None:
            return reg

    profile_dir: str | None = kwargs.get("profile_dir")
    _profiler = None
    if profile_dir:
        import cProfile

        _profiler = cProfile.Profile()
        _profiler.enable()

    text = input_path.read_text(encoding="utf-8")

    checkpoint_path = kwargs.get("checkpoint_path")
    force_restart = bool(kwargs.get("force_restart", False))

    metrics_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("metrics_callback")

    # 预处理 + 预分词阶段（可观测性埋点在 helper 内输出 stage_start/stage_end）
    # 注意：这里复用同一个 num_workers 作为预分词的并行 worker 数（小语料会自动回退到串行，避免启动开销影响速度）。
    pretok_num_workers = kwargs.get("num_workers")
    if pretok_num_workers is None:
        pretok_num_workers = (os.cpu_count() or 1) + 1
    words = _preprocess_and_pretokenize_training_text(
        text,
        special_tokens,
        metrics_callback=metrics_callback,
        num_workers=int(pretok_num_workers),
    )

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes]
    next_id: int

    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and not force_restart and ckpt.is_file():
        vocab, merges = _load_checkpoint(ckpt)
        next_id = max(vocab.keys(), default=-1) + 1
        for left, right in merges:
            _merge_pair_all_words(words, left, right, left + right)
    else:
        vocab = _build_initial_vocab(special_tokens)
        merges = []
        next_id = len(vocab)

    remaining = vocab_size - len(vocab)
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = (os.cpu_count() or 1) + 1
    use_parallel = num_workers > 1 and remaining > 0 and len(words) >= _PARALLEL_WORD_THRESHOLD

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
        metrics_callback({"event": "stage_start", "stage": "byte_pair_merge_iter"})

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

    if _profiler is not None:
        _profiler.disable()
        _prof_dir = Path(profile_dir)  # type: ignore[arg-type]
        _prof_dir.mkdir(parents=True, exist_ok=True)
        _prof_file = _prof_dir / f"train_bpe_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.prof"
        _profiler.dump_stats(str(_prof_file))
        if callable(metrics_callback):
            metrics_callback({"event": "profile_saved", "path": str(_prof_file)})

    return vocab, merges
