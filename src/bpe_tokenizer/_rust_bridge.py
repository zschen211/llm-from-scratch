"""BPE 核心：通过 PyO3 扩展 `bpe_core` 提供实现；必须已安装该 Rust 扩展。

未安装时导入本模块会失败，请在 `libs/bpe_core` 下执行：
  maturin develop --release
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_dispatch_logged: set[str] = set()


def _log_backend_once(op: str) -> None:
    if op in _dispatch_logged:
        return
    _dispatch_logged.add(op)
    _log.info("[_rust_bridge] 首次调用 %s -> Rust (bpe_core)", op)


_BPE_CORE_HINT = (
    "无法导入 bpe_core（Rust 扩展）。请在仓库根目录执行："
    "cd libs/bpe_core && uv run maturin develop --release"
    "（或 pip install maturin 后在该目录执行 maturin develop --release）。"
)

try:
    from bpe_core import (
        WordsChunkWithIndex as _RustWordsChunkWithIndex,
        count_pairs_py as rust_count_pairs,
        dump_words_chunk_py as rust_dump_words_chunk,
        load_words_chunk_py as rust_load_words_chunk,
        merge_pair_all_words_with_deltas_py as rust_merge_pair_all_words_with_deltas,
        preprocess_and_pretokenize_py as rust_preprocess_and_pretokenize,
        pretokenize_with_pat_py as rust_pretokenize_with_pat,
        train_bpe_full_py as rust_train_bpe_full,
    )
except ImportError as e:
    raise ImportError(_BPE_CORE_HINT) from e

RUST_AVAILABLE = True
RUST_PAT_AVAILABLE = True

_log.info("[_rust_bridge] bpe_core 已加载，本模块仅使用 Rust 实现")


def count_pairs(words: list[list[bytes]], min_freq: int = 1) -> dict[tuple[bytes, bytes], int]:
    """统计 words 中所有相邻字节对的频率（Rust）。"""
    _log_backend_once("count_pairs")
    return rust_count_pairs(words, min_freq)


def merge_pair_all_words_with_deltas(
    words: list[list[bytes]],
    left: bytes,
    right: bytes,
    merged: bytes,
) -> dict[tuple[bytes, bytes], int]:
    """将所有 words 内的 (left, right) 合并为 merged，并返回频率增量（Rust）。"""
    _log_backend_once("merge_pair_all_words_with_deltas")
    return rust_merge_pair_all_words_with_deltas(words, left, right, merged)


def preprocess_and_pretokenize(
    text: str,
    special_tokens: list[str],
) -> list[list[bytes]]:
    """预处理和预分词（Rust）。"""
    _log_backend_once("preprocess_and_pretokenize")
    return rust_preprocess_and_pretokenize(text, special_tokens)


def pretokenize_with_pat(
    text: str,
    special_tokens: list[str],
    use_tiktoken_pat: bool = True,
) -> list[list[bytes]]:
    """使用 Rust fancy-regex 的预分词（Rust）。

    Args:
        text: 输入文本
        special_tokens: 特殊 token 列表
        use_tiktoken_pat: 是否使用 tiktoken GPT-2 PAT（默认 True），否则使用 CS336 PAT
    """
    _log_backend_once("pretokenize_with_pat")
    return rust_pretokenize_with_pat(text, special_tokens, use_tiktoken_pat)


class WordsChunkWithIndex:
    """带倒排索引的 words chunk（Rust `bpe_core.WordsChunkWithIndex` 包装）。"""

    def __init__(self, words: list[list[bytes]]):
        _log_backend_once("WordsChunkWithIndex.__init__")
        self._impl = _RustWordsChunkWithIndex(words)

    def build_index(self) -> None:
        self._impl.build_index()

    @property
    def _index_built(self) -> bool:
        """与 `_merge_optimizer.count_pairs_with_index` 兼容：Rust 侧 merge 时会按需建索引。"""
        return True

    def merge_pair_with_deltas(
        self, left: bytes, right: bytes, merged: bytes
    ) -> dict[tuple[bytes, bytes], int]:
        return self._impl.merge_pair_with_deltas(left, right, merged)

    @property
    def words(self) -> list[list[bytes]]:
        return self._impl.get_words()

    def save(self, path: str | Path) -> None:
        """使用与 `dump_words_chunk` 相同的 bincode 格式落盘。"""
        rust_dump_words_chunk(str(path), self.words)

    @classmethod
    def load(cls, path: str | Path) -> WordsChunkWithIndex:
        words = rust_load_words_chunk(str(path))
        chunk = cls(words)
        chunk.build_index()
        return chunk


def dump_words_chunk(path: str, words: list[list[bytes]]) -> None:
    """保存 words chunk（Rust bincode）。"""
    _log_backend_once("dump_words_chunk")
    rust_dump_words_chunk(path, words)


def load_words_chunk(path: str) -> list[list[bytes]]:
    """加载 words chunk（Rust bincode）。"""
    _log_backend_once("load_words_chunk")
    return rust_load_words_chunk(path)


def train_bpe_full(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 4,
    stream_chunk_chars: int = 1_000_000,
    chunks_dir: str = ".bpe_chunks",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """完整 BPE 训练（Rust）。"""
    _log_backend_once("train_bpe_full")
    return rust_train_bpe_full(
        input_path,
        vocab_size,
        special_tokens,
        num_workers,
        stream_chunk_chars,
        chunks_dir,
    )


__all__ = [
    "WordsChunkWithIndex",
    "count_pairs",
    "dump_words_chunk",
    "load_words_chunk",
    "merge_pair_all_words_with_deltas",
    "preprocess_and_pretokenize",
    "pretokenize_with_pat",
    "train_bpe_full",
    "RUST_AVAILABLE",
    "RUST_PAT_AVAILABLE",
]
