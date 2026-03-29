"""BPE 训练：预处理与预分词（含流式读入与 chunk 落盘）。"""

from __future__ import annotations

import logging
import multiprocessing as _mp
import os
import time
from pathlib import Path
from typing import Any, Callable

from ._pat import ENCODE_SPLIT_PATTERN

_log = logging.getLogger(__name__)

_PRETOK_PARALLEL_SEGMENTS_THRESHOLD = 5000


def _pretok_worker_main(
    special_set: set[str],
    cmd_q: _mp.Queue,  # type: ignore[type-arg]
    result_q: _mp.Queue,  # type: ignore[type-arg]
) -> None:
    """
    持久化 worker：对“预分词分块”执行 PAT findall，并输出该分块对应的 words[] 片段。
    """
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


def _preprocess_and_pretokenize_training_text(
    text: str,
    special_tokens: list[str],
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    num_workers: int | None = None,
    use_rust: bool = True,
) -> list[list[bytes]]:
    """
    把预处理与预分词拆成两个阶段，便于输出可观测性指标：
    - 数据预处理：按特殊串切分，得到一段段文本/特殊 token 段
    - 数据预分词：对普通段做 PAT findall，并把特殊 token 作为原子 token

    预分词必须与 tiktoken/GPT-2 的 PAT 一致；当前 bpe_core 使用 Rust `regex` 库，
    无法编译含环视的 PAT（如 `(?!\\S)`），故此处始终使用本文件的 Python 实现。
    流式落盘时的倒排 chunk 仍通过 `_rust_bridge.WordsChunkWithIndex` 使用 Rust（若已安装 bpe_core）。

    Args:
        text: 输入文本
        special_tokens: 特殊 token 列表
        metrics_callback: 性能指标回调函数
        num_workers: 并行 worker 数量
        use_rust: 是否使用 Rust 实现（默认 True）
    """
    # 尝试使用 Rust 实现
    if use_rust:
        try:
            from ._rust_bridge import RUST_PAT_AVAILABLE, pretokenize_with_pat

            if RUST_PAT_AVAILABLE:
                if callable(metrics_callback):
                    metrics_callback({"event": "stage_start", "stage": "data_pretokenization"})

                pretoken_t0 = time.perf_counter()
                words = pretokenize_with_pat(text, special_tokens, use_tiktoken_pat=True)
                pretok_ms = 1000.0 * (time.perf_counter() - pretoken_t0)

                if callable(metrics_callback):
                    metrics_callback(
                        {
                            "event": "stage_end",
                            "stage": "data_pretokenization",
                            "metrics": {"pretok_count": len(words), "pretok_ms": pretok_ms},
                        }
                    )

                return words
        except Exception as e:
            _log.warning(f"Rust PAT 分词失败，回退到 Python 实现: {e}")

    # Python 实现（原有逻辑）
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
        task_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]
        res_q: _mp.Queue = _mp.Queue()  # type: ignore[type-arg]

        workers: list[_mp.Process] = []
        for _ in range(num_workers or 0):
            p = _mp.Process(target=_pretok_worker_main, args=(special_set, task_q, res_q), daemon=True)
            p.start()
            workers.append(p)

        n = len(segments)
        chunk_size = max(1, (n + len(workers) - 1) // len(workers))
        chunk_tasks: int = 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            task_q.put((chunk_tasks, segments[start:end]))
            chunk_tasks += 1

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

        return sorted(set(boundaries))


def streaming_pretokenize_to_chunk_files(
    input_path: Path,
    special_tokens: list[str],
    chunks_dir: Path,
    *,
    stream_chunk_chars: int,
    memory_target_percent: float,
    use_inverted_index: bool,
    metrics_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[Path], int, float]:
    """
    顺序读入语料、按 special 对齐块、预分词并视内存阈值将 words 落盘为 chunk pkl。

    返回 (chunk_files, pretok_total_count, pretok_elapsed_ms)。
    """
    from .train_bpe_merge import (
        _dump_words_chunk,
        _dump_words_chunk_with_index,
        _force_release_memory,
        _get_system_memory_percent,
    )

    pretok_dump_threshold = max(30.0, memory_target_percent - 20.0)
    split_special = max(special_tokens, key=len, default="<|endoftext|>")
    file_size = input_path.stat().st_size

    if callable(metrics_callback):
        metrics_callback({"event": "stage_start", "stage": "data_pretokenization"})
    _log.info(
        "[pretokenize] streaming pretokenization starting (chunk_chars=%d, mem_threshold=%.0f%%)",
        stream_chunk_chars,
        pretok_dump_threshold,
    )
    pretok_total_t0 = time.perf_counter()
    pretok_total_count = 0
    chunk_index = 0
    accumulated_words: list[list[bytes]] = []
    chunk_files: list[Path] = []

    with open(input_path, "r", encoding="utf-8") as f:
        carry = ""

        while True:
            if accumulated_words:
                mem = _get_system_memory_percent()
                if mem is not None and mem >= pretok_dump_threshold:
                    chunk_index += 1
                    cpath = chunks_dir / f"chunk_{chunk_index:06d}.pkl"

                    if use_inverted_index:
                        from ._rust_bridge import WordsChunkWithIndex

                        chunk_with_index = WordsChunkWithIndex(accumulated_words)
                        index_t0 = time.perf_counter()
                        chunk_with_index.build_index()
                        index_ms = 1000.0 * (time.perf_counter() - index_t0)
                        _pi = getattr(chunk_with_index, "pair_index", None)
                        if _pi is not None:
                            _log.info(
                                "[pretokenize] memory %.0f%% >= threshold, spilled chunk_%06d (words=%d, index_ms=%.1f, pairs=%d)",
                                mem,
                                chunk_index,
                                len(accumulated_words),
                                index_ms,
                                len(_pi),
                            )
                        else:
                            _log.info(
                                "[pretokenize] memory %.0f%% >= threshold, spilled chunk_%06d (words=%d, index_ms=%.1f, rust_index)",
                                mem,
                                chunk_index,
                                len(accumulated_words),
                                index_ms,
                            )
                        _dump_words_chunk_with_index(cpath, chunk_with_index)
                    else:
                        _dump_words_chunk(cpath, accumulated_words)
                        _log.info(
                            "[pretokenize] memory %.0f%% >= threshold, spilled chunk_%06d to disk (words=%d)",
                            mem,
                            chunk_index,
                            len(accumulated_words),
                        )

                    chunk_files.append(cpath)
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

        if use_inverted_index:
            from ._rust_bridge import WordsChunkWithIndex

            chunk_with_index = WordsChunkWithIndex(accumulated_words)
            _log.info(
                "[pretokenize] building inverted index for chunk_%06d (words=%d)",
                chunk_index,
                len(accumulated_words),
            )
            index_t0 = time.perf_counter()
            chunk_with_index.build_index()
            index_ms = 1000.0 * (time.perf_counter() - index_t0)
            _pi = getattr(chunk_with_index, "pair_index", None)
            if _pi is not None:
                _log.info("[pretokenize] index built in %.1fms, pairs=%d", index_ms, len(_pi))
            else:
                _log.info("[pretokenize] index built in %.1fms (rust_index)", index_ms)
            _dump_words_chunk_with_index(cpath, chunk_with_index)
        else:
            _dump_words_chunk(cpath, accumulated_words)

        chunk_files.append(cpath)
        accumulated_words = []

    pretok_elapsed_ms = 1000.0 * (time.perf_counter() - pretok_total_t0)
    _log.info(
        "[pretokenize] done: total_words=%d, chunks_on_disk=%d, elapsed=%.1fs",
        pretok_total_count,
        len(chunk_files),
        pretok_elapsed_ms / 1000.0,
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

    return chunk_files, pretok_total_count, pretok_elapsed_ms
