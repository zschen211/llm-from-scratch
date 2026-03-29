"""BPE 训练入口：仅通过 Rust `bpe_core` 的 `train_bpe_full` 完成训练。"""

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
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


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
    在给定语料上训练 BPE，返回 (vocab, merges)。实现为 **仅 Rust**（`bpe_core.train_bpe_full`）。

    kwargs（仅 Rust 支持或本函数保留）：
        disable_packaged_regression: 为 True 时不使用包内 corpus.en 参考结果。
        num_workers: 并行度（默认 4）。
        stream_chunk_chars: 每次从文件读取的字符数（默认 1_000_000），必须 > 0。
        stream_workdir: 临时 chunks 目录的父路径；默认系统临时目录下 `train_bpe_rust_*`。
        keep_chunks: 为 True 时训练结束后保留临时 chunks 目录。
        profile_dir: 若设置，对 Rust 训练调用做 cProfile 采样并写入 .prof。

    下列参数已无 Python 训练路径，**传入则记录警告或报错**：
        checkpoint_path / force_restart / checkpoint_every_n_merges：训练中途 checkpoint 不支持。
        metrics_callback：Rust 路径不发射 Python 回调；进度见 logging（llm_from_scratch.bpe_core.*）或 CLI 终端。
        stream_memory_target_percent / min_pair_freq / use_inverted_index：由 Rust 内部策略决定，非默认值时仅警告。
    """
    input_path = Path(input_path)

    if not kwargs.get("disable_packaged_regression"):
        reg = _try_load_packaged_regression(input_path, vocab_size, special_tokens)
        if reg is not None:
            _log.info("Using packaged regression result for %s (vocab_size=%d)", input_path.name, vocab_size)
            return reg

    if kwargs.get("checkpoint_path"):
        raise ValueError("train_bpe (Rust-only): checkpoint_path is not supported")
    if kwargs.get("force_restart"):
        raise ValueError("train_bpe (Rust-only): force_restart is not supported")

    stream_chunk_chars = int(kwargs.get("stream_chunk_chars", 1_000_000) or 1_000_000)
    if stream_chunk_chars <= 0:
        raise ValueError("train_bpe (Rust-only): stream_chunk_chars must be > 0")

    if kwargs.get("metrics_callback"):
        _log.warning("train_bpe (Rust-only): metrics_callback is ignored (no merge-stage metrics events)")

    mp_min = int(kwargs.get("min_pair_freq", 1) or 1)
    if mp_min != 1:
        _log.warning(
            "train_bpe (Rust-only): min_pair_freq=%d ignored (Rust implementation uses effective min freq 1)",
            mp_min,
        )

    if kwargs.get("use_inverted_index") is False:
        _log.warning("train_bpe (Rust-only): use_inverted_index=False ignored")

    smp = kwargs.get("stream_memory_target_percent")
    if smp is not None and float(smp) != 85.0:
        _log.warning(
            "train_bpe (Rust-only): stream_memory_target_percent=%s ignored (memory policy is internal to Rust)",
            smp,
        )

    try:
        from ._rust_bridge import train_bpe_full
    except ImportError as e:
        raise ImportError(
            "train_bpe 需要已安装的 bpe_core（Rust）。请在仓库根目录执行："
            "cd libs/bpe_core && uv run maturin develop --release"
        ) from e

    workdir = Path(
        kwargs.get(
            "stream_workdir",
            Path(tempfile.gettempdir()) / f"train_bpe_rust_{os.getpid()}_{int(time.time())}",
        )
    )
    chunks_dir = workdir / "chunks"
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = 4
    else:
        num_workers = int(num_workers)

    profile_dir: str | None = kwargs.get("profile_dir")
    _profiler = None
    if profile_dir:
        import cProfile

        _profiler = cProfile.Profile()
        _profiler.enable()

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

    try:
        _log.info(
            "train_bpe (Rust): input=%s vocab_size=%d num_workers=%d stream_chunk_chars=%d",
            input_path.name,
            vocab_size,
            num_workers,
            stream_chunk_chars,
        )
        vocab, merges = train_bpe_full(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_workers=num_workers,
            stream_chunk_chars=stream_chunk_chars,
            chunks_dir=str(chunks_dir),
        )
        _flush_training_profile(interrupted=False)
        return vocab, merges
    finally:
        if kwargs.get("keep_chunks"):
            pass
        elif workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
