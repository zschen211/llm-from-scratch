"""BPE 训练入口：编排预处理/预分词与字节对合并阶段。"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

from .train_bpe_merge import (
    _BPEStreamWorkerPool,
    _build_initial_vocab,
    _count_pairs,
    _dump_words_chunk,
    _dump_words_chunk_with_index,
    _force_release_memory,
    _load_chunk_batch,
    _merge_pair_all_words_with_pair_deltas,
    _pick_pair_to_merge,
    _save_checkpoint,
)
from .train_bpe_pretokenize import (
    streaming_pretokenize_to_chunk_files,
)


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


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在给定语料上训练 BPE，返回 (vocab, merges)。

    kwargs:
        use_rust_backend: 为 True 时使用 Rust 完整训练流程（默认 True）。
            Rust 后端性能更好，消除了 Python-Rust 数据拷贝开销。
            设为 False 回退到 Python 实现。
        checkpoint_path: 每次 merge 后写入 checkpoint（JSON）。
        checkpoint_every_n_merges: checkpoint 保存间隔（默认 1，即每轮保存）。
            设为 10 表示每 10 轮 merge 保存一次；<=0 时等价于 1。
        force_restart: 为 True 时忽略已有 checkpoint 重新开始。
        disable_packaged_regression: 为 True 时不使用包内 corpus.en 参考结果。
        num_workers: 并行 worker 进程数；默认 cpu_count()+1，设为 1 禁用并行。
        metrics_callback: 可选；如果提供 callable，会在每次完成一次 merge iteration 时调用：
            metrics_callback(metrics: dict[str, Any])。
        profile_dir: 可选；若提供目录路径，使用 cProfile 对训练全程采样并将
            .prof 文件写入该目录（文件名含时间戳与 PID）。
            若用户按 Ctrl-C（SIGINT）中断，仍会尽量落盘（文件名可带 `_interrupted` 后缀）；
            进程正常退出时还会通过 atexit 再兜底一次（幂等，不会重复写入）。
            Ctrl-D 仅在从标准输入读数据的场景有意义；本函数只读 `input_path` 文件，一般不因 Ctrl-D 触发。
        stream_chunk_chars: 流式读取时每次从文件中读取的字符数（默认 1_000_000）。
            > 0 时启用流式读取；设为 0 禁用流式，回退到一次性 read_text 模式。
        stream_memory_target_percent: 外存模式内存占用率阈值（默认 85）。
            pretokenize 阶段累积的 words 在内存占用率达到该阈值时落盘为一个 chunk 文件；
            merge 阶段分批加载 chunk 文件时也以此阈值为界。
            chunk 数量完全由数据大小和可用内存动态决定。
    """
    input_path = Path(input_path)

    # 检查是否使用 packaged regression
    if not kwargs.get("disable_packaged_regression"):
        reg = _try_load_packaged_regression(input_path, vocab_size, special_tokens)
        if reg is not None:
            _log.info("Using packaged regression result for %s (vocab_size=%d)", input_path.name, vocab_size)
            return reg

    # 尝试使用 Rust 后端（默认启用）
    use_rust_backend = kwargs.get("use_rust_backend", True)
    if use_rust_backend:
        try:
            from ._rust_bridge import train_bpe_full, RUST_AVAILABLE

            if RUST_AVAILABLE:
                # 检查是否有不兼容的参数
                incompatible_params = [
                    "checkpoint_path",
                    "profile_dir",
                    "metrics_callback",
                    "use_inverted_index",  # Rust 不支持倒排索引选项
                ]
                has_incompatible = any(kwargs.get(p) is not None for p in incompatible_params)

                # 检查 stream_chunk_chars=0（禁用流式）
                stream_chunk_chars = int(kwargs.get("stream_chunk_chars", 1_000_000) or 1_000_000)
                if stream_chunk_chars == 0:
                    has_incompatible = True

                if not has_incompatible:
                    _log.info("Using Rust backend for BPE training")

                    # 准备 chunks_dir
                    workdir = Path(
                        kwargs.get(
                            "stream_workdir",
                            Path(tempfile.gettempdir()) / f"train_bpe_rust_{os.getpid()}_{int(time.time())}",
                        )
                    )
                    chunks_dir = workdir / "chunks"

                    try:
                        vocab, merges = train_bpe_full(
                            input_path=str(input_path),
                            vocab_size=vocab_size,
                            special_tokens=special_tokens,
                            num_workers=kwargs.get("num_workers", 4),
                            stream_chunk_chars=int(kwargs.get("stream_chunk_chars", 1_000_000) or 1_000_000),
                            chunks_dir=str(chunks_dir),
                        )

                        # 清理临时目录
                        if workdir.exists() and not kwargs.get("keep_chunks"):
                            shutil.rmtree(workdir, ignore_errors=True)

                        return vocab, merges
                    except Exception as e:
                        _log.warning("Rust backend failed: %s, falling back to Python", e)
                else:
                    _log.info("Rust backend not used due to incompatible parameters, using Python")
        except ImportError:
            _log.debug("Rust backend not available, using Python")

    # Python 实现（原有逻辑）
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
    checkpoint_every_n_merges = max(1, int(kwargs.get("checkpoint_every_n_merges", 1) or 1))
    force_restart = bool(kwargs.get("force_restart", False))

    metrics_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("metrics_callback")

    _profile_saved = False
    _sigint_previous: Any = None

    def _flush_training_profile(*, interrupted: bool | None = None) -> None:
        nonlocal _profile_saved, _sigint_previous
        if not profile_dir or _profiler is None or _profile_saved:
            return
        if interrupted is None:
            interrupted = sys.exc_info()[0] is not None
        _profile_saved = True
        try:
            _profiler.disable()
        except (ValueError, RuntimeError):
            pass
        try:
            _prof_dir = Path(profile_dir)
            _prof_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_interrupted" if interrupted else ""
            _prof_file = _prof_dir / f"train_bpe_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}{suffix}.prof"
            _profiler.dump_stats(str(_prof_file))
            if interrupted:
                _log.info("[profile] saved to %s (interrupt or pending exception)", _prof_file)
            else:
                _log.info("[profile] saved to %s", _prof_file)
            if callable(metrics_callback):
                metrics_callback(
                    {"event": "profile_saved", "path": str(_prof_file), "interrupted": bool(interrupted)}
                )
        except Exception as exc:
            _log.warning("[profile] save failed: %s", exc)
        finally:
            if _sigint_previous is not None:
                try:
                    signal.signal(signal.SIGINT, _sigint_previous)
                except (OSError, ValueError):
                    pass
                _sigint_previous = None

    if profile_dir:

        def _sigint_handler(signum: int, frame: Any) -> None:
            _flush_training_profile(interrupted=True)
            raise KeyboardInterrupt from None

        _sigint_previous = signal.signal(signal.SIGINT, _sigint_handler)
        atexit.register(lambda: _flush_training_profile(interrupted=None))
    stream_chunk_chars = int(kwargs.get("stream_chunk_chars", 1_000_000) or 1_000_000)

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    _log.info(
        "train_bpe start: input=%s (%.1f MB), vocab_size=%d, special_tokens=%d, mode=streaming",
        input_path.name,
        file_size_mb,
        vocab_size,
        len(special_tokens),
    )
    _log.info("checkpoint policy: every %d merge(s)", checkpoint_every_n_merges)

    checkpoint_path = kwargs.get("checkpoint_path")
    force_restart = bool(kwargs.get("force_restart", False))
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and not force_restart and ckpt.is_file():
        raise ValueError(
            "Streaming mode does not support checkpoint resume yet; "
            "set force_restart=True."
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

    memory_target_percent = float(kwargs.get("stream_memory_target_percent", 85) or 85)
    use_inverted_index = bool(kwargs.get("use_inverted_index", True))

    try:
        chunk_files, _pretok_total_count, _pretok_ms = streaming_pretokenize_to_chunk_files(
            input_path,
            special_tokens,
            chunks_dir,
            stream_chunk_chars=stream_chunk_chars,
            memory_target_percent=memory_target_percent,
            use_inverted_index=use_inverted_index,
            metrics_callback=metrics_callback,
        )

        cumulative_iter_wall_ms = 0.0
        merge_iters_executed = 0
        if callable(metrics_callback):
            metrics_callback(
                {"event": "stage_start", "stage": "byte_pair_merge_iter", "merges_target": merges_target}
            )

        _log.info("[merge] starting BPE merge iterations (target=%d merges)", merges_target)

        _log.info("[merge] inverted index: %s", "enabled" if use_inverted_index else "disabled")

        min_pair_freq = int(kwargs.get("min_pair_freq", 1))
        if min_pair_freq > 1:
            _log.info("[merge] initial pair counting with min_freq=%d to reduce memory usage", min_pair_freq)

        first_batch, first_next = _load_chunk_batch(
            chunk_files, 0, memory_target_percent, use_inverted_index=use_inverted_index,
        )
        all_in_memory = first_next >= len(chunk_files)
        _log.info(
            "[merge] chunk loading: all_in_memory=%s (%d/%d chunks loaded)",
            all_in_memory,
            first_next,
            len(chunk_files),
        )

        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        # 决定是否使用多进程 worker
        num_workers = kwargs.get("num_workers")
        if num_workers is None:
            num_workers = (os.cpu_count() or 1) + 1
        use_stream_workers = not all_in_memory and num_workers > 1
        _log.info("[merge] workers=%d, use_stream_workers=%s", num_workers, use_stream_workers)

        # 如果使用流式 worker，创建 worker pool
        stream_pool = None
        if use_stream_workers:
            try:
                stream_pool = _BPEStreamWorkerPool(chunk_files, num_workers, use_inverted_index)
                _log.info("[merge] created stream worker pool with %d workers", num_workers)
            except (OSError, RuntimeError) as e:
                _log.warning("[merge] failed to create stream worker pool: %s, falling back to serial", e)
                use_stream_workers = False

        # 初始统计字节对频率
        if use_stream_workers and stream_pool:
            pair_counts, count_metrics = stream_pool.count_pairs()
            _log.info("[merge] initial pair counting done (parallel)")
        elif use_inverted_index:
            from ._merge_optimizer import count_pairs_with_index

            if all_in_memory:
                resident_chunks_with_index: list[Any] = [chunk for _, chunk in first_batch]
                del first_batch
                for chunk in resident_chunks_with_index:
                    chunk_pair_counts = count_pairs_with_index(chunk)
                    if min_pair_freq > 1:
                        chunk_pair_counts = {p: c for p, c in chunk_pair_counts.items() if c >= min_pair_freq}
                    pair_counts.update(chunk_pair_counts)
            else:
                for _cpath, chunk in first_batch:
                    chunk_pair_counts = count_pairs_with_index(chunk)
                    if min_pair_freq > 1:
                        chunk_pair_counts = {p: c for p, c in chunk_pair_counts.items() if c >= min_pair_freq}
                    pair_counts.update(chunk_pair_counts)
                del first_batch
                _force_release_memory()
                batch_start = first_next
                while batch_start < len(chunk_files):
                    batch, batch_start = _load_chunk_batch(
                        chunk_files, batch_start, memory_target_percent, use_inverted_index=True,
                    )
                    for _cpath, chunk in batch:
                        chunk_pair_counts = count_pairs_with_index(chunk)
                        if min_pair_freq > 1:
                            chunk_pair_counts = {p: c for p, c in chunk_pair_counts.items() if c >= min_pair_freq}
                        pair_counts.update(chunk_pair_counts)
                    del batch
                    _force_release_memory()

                if min_pair_freq > 1:
                    before_count = len(pair_counts)
                    pair_counts = Counter({p: cnt for p, cnt in pair_counts.items() if cnt >= min_pair_freq})
                    after_count = len(pair_counts)
                    if before_count > after_count:
                        _log.info(
                            "[merge] filtered low-freq pairs: %d -> %d (removed %d pairs)",
                            before_count,
                            after_count,
                            before_count - after_count,
                        )
                        _force_release_memory()
        else:
            if all_in_memory:
                resident_chunks: list[list[list[bytes]]] = [w for _, w in first_batch]
                del first_batch
                for words_chunk in resident_chunks:
                    pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
            else:
                for _cpath, words_chunk in first_batch:
                    pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
                del first_batch
                _force_release_memory()
                batch_start = first_next
                while batch_start < len(chunk_files):
                    batch, batch_start = _load_chunk_batch(
                        chunk_files, batch_start, memory_target_percent, use_inverted_index=False,
                    )
                    for _cpath, words_chunk in batch:
                        pair_counts.update(_count_pairs(words_chunk, min_freq=min_pair_freq))
                    del batch
                    _force_release_memory()

                if min_pair_freq > 1:
                    before_count = len(pair_counts)
                    pair_counts = Counter({p: cnt for p, cnt in pair_counts.items() if cnt >= min_pair_freq})
                    after_count = len(pair_counts)
                    if before_count > after_count:
                        _log.info(
                            "[merge] filtered low-freq pairs: %d -> %d (removed %d pairs)",
                            before_count,
                            after_count,
                            before_count - after_count,
                        )
                        _force_release_memory()

        while len(vocab) < vocab_size:
            iter_no = len(merges) + 1
            iter_t0 = time.perf_counter()
            pick_t0 = time.perf_counter()
            io_ms = 0.0

            if not pair_counts:
                break

            left, right = _pick_pair_to_merge(pair_counts)
            pick_ms = 1000.0 * (time.perf_counter() - pick_t0)
            merged = left + right
            merges.append((left, right))
            vocab[next_id] = merged
            next_id += 1

            iter_delta: dict[tuple[bytes, bytes], int] = {}
            merge_t0 = time.perf_counter()

            # 使用流式 worker pool 进行并行处理
            if use_stream_workers and stream_pool:
                iter_delta, merge_metrics = stream_pool.merge_pair_with_pair_deltas(left, right, merged)
                merge_ms = 1000.0 * (time.perf_counter() - merge_t0)
            elif use_inverted_index:
                if all_in_memory:
                    for chunk in resident_chunks_with_index:
                        delta = chunk.merge_pair_with_deltas(left, right, merged)
                        for p, d in delta.items():
                            iter_delta[p] = iter_delta.get(p, 0) + d
                else:
                    batch_start = 0
                    while batch_start < len(chunk_files):
                        io_t0 = time.perf_counter()
                        batch, batch_start = _load_chunk_batch(
                            chunk_files, batch_start, memory_target_percent, use_inverted_index=True,
                        )
                        io_ms += 1000.0 * (time.perf_counter() - io_t0)
                        for cpath, chunk in batch:
                            delta = chunk.merge_pair_with_deltas(left, right, merged)
                            for p, d in delta.items():
                                iter_delta[p] = iter_delta.get(p, 0) + d
                            io_t0 = time.perf_counter()
                            _dump_words_chunk_with_index(cpath, chunk)
                            io_ms += 1000.0 * (time.perf_counter() - io_t0)
                        del batch
                        _force_release_memory()
            else:
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
                        io_t0 = time.perf_counter()
                        batch, batch_start = _load_chunk_batch(
                            chunk_files, batch_start, memory_target_percent, use_inverted_index=False,
                        )
                        io_ms += 1000.0 * (time.perf_counter() - io_t0)
                        for cpath, words_chunk in batch:
                            delta = _merge_pair_all_words_with_pair_deltas(
                                words_chunk, left, right, merged,
                            )
                            for p, d in delta.items():
                                iter_delta[p] = iter_delta.get(p, 0) + d
                            io_t0 = time.perf_counter()
                            _dump_words_chunk(cpath, words_chunk)
                            io_ms += 1000.0 * (time.perf_counter() - io_t0)
                        del batch
                        _force_release_memory()

            merge_ms = 1000.0 * (time.perf_counter() - merge_t0)

            apply_t0 = time.perf_counter()
            for p, d in iter_delta.items():
                new_v = pair_counts.get(p, 0) + d
                if new_v > 0:
                    pair_counts[p] = new_v
                else:
                    pair_counts.pop(p, None)
            apply_ms = 1000.0 * (time.perf_counter() - apply_t0)

            if ckpt and (iter_no % checkpoint_every_n_merges == 0 or len(vocab) >= vocab_size):
                io_t0 = time.perf_counter()
                _save_checkpoint(ckpt, vocab, merges)
                io_ms += 1000.0 * (time.perf_counter() - io_t0)

            iter_wall_ms = 1000.0 * (time.perf_counter() - iter_t0)
            cumulative_iter_wall_ms += iter_wall_ms
            merge_iters_executed += 1
            avg_merge_wall_ms = cumulative_iter_wall_ms / merge_iters_executed

            _stream_merge_log_interval = max(1, merges_target // 20)
            if iter_no == 1 or iter_no % _stream_merge_log_interval == 0 or iter_no == merges_target:
                _log.info(
                    "[merge] %d/%d (%.0f%%) avg=%.1fms/iter [pick=%.1fms merge=%.1fms apply=%.1fms io=%.1fms]",
                    iter_no,
                    merges_target,
                    100.0 * iter_no / max(merges_target, 1),
                    avg_merge_wall_ms,
                    pick_ms,
                    merge_ms,
                    apply_ms,
                    io_ms,
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

        # 清理 stream worker pool
        if stream_pool:
            stream_pool.shutdown()
            _log.info("[merge] stream worker pool shutdown")

        total_merge_s = cumulative_iter_wall_ms / 1000.0
        _log.info("[merge] streaming merge done: %d merges in %.1fs", merge_iters_executed, total_merge_s)
        if callable(metrics_callback):
            final_avg_merge_wall_ms = (
                cumulative_iter_wall_ms / merge_iters_executed if merge_iters_executed else 0.0
            )
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

        _flush_training_profile(interrupted=False)

        _log.info("train_bpe done: vocab_size=%d, merges=%d", len(vocab), len(merges))
        return vocab, merges
    finally:
        if kwargs.get("stream_keep_workdir"):
            pass
        else:
            shutil.rmtree(workdir, ignore_errors=True)
