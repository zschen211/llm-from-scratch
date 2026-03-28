"""BPE 核心模块的 Rust 加速版本。

这个模块提供了与 Python 实现相同的接口，但使用 Rust 实现以获得更好的性能。
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# 各入口首次实际调用时打一条 INFO，便于确认运行时是否走到 Rust（需配置 logging 级别）。
_dispatch_logged: set[str] = set()


def _log_backend_once(op: str, *, rust: bool) -> None:
    backend = "Rust (bpe_core)" if rust else "Python"
    key = f"{op}:{backend}"
    if key in _dispatch_logged:
        return
    _dispatch_logged.add(key)
    _log.info("[_rust_bridge] 首次调用 %s -> %s", op, backend)


try:
    # 尝试导入 Rust 实现
    from bpe_core import (
        WordsChunkWithIndex as RustWordsChunkWithIndex,
        count_pairs_py as rust_count_pairs,
        merge_pair_all_words_with_deltas_py as rust_merge_pair_all_words_with_deltas,
        preprocess_and_pretokenize_py as rust_preprocess_and_pretokenize,
    )

    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    RustWordsChunkWithIndex = None
    rust_count_pairs = None
    rust_merge_pair_all_words_with_deltas = None
    rust_preprocess_and_pretokenize = None
    _log.debug("bpe_core 导入失败，将使用 Python 回退: %s", e)

# 如果 Rust 不可用，回退到 Python 实现
if not RUST_AVAILABLE:
    from ._merge_optimizer import WordsChunkWithIndex as PythonWordsChunkWithIndex
    from .train_bpe_merge import (
        _count_pairs as python_count_pairs,
        _merge_pair_all_words_with_pair_deltas as python_merge_pair_all_words_with_deltas,
    )
    from .train_bpe_pretokenize import (
        _preprocess_and_pretokenize_training_text as python_preprocess_and_pretokenize,
    )

if RUST_AVAILABLE:
    _log.info("[_rust_bridge] bpe_core 已加载，本模块优先使用 Rust")
else:
    _log.info("[_rust_bridge] bpe_core 不可用，本模块使用 Python 实现")


def count_pairs(words: list[list[bytes]], min_freq: int = 1) -> dict[tuple[bytes, bytes], int]:
    """统计 words 中所有相邻字节对的频率。

    优先使用 Rust 实现，如果不可用则回退到 Python 实现。
    """
    _log_backend_once("count_pairs", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        return rust_count_pairs(words, min_freq)
    else:
        return python_count_pairs(words, min_freq)


def merge_pair_all_words_with_deltas(
    words: list[list[bytes]],
    left: bytes,
    right: bytes,
    merged: bytes,
) -> dict[tuple[bytes, bytes], int]:
    """将所有 words 内的 (left, right) 合并为 merged，并返回频率增量。

    优先使用 Rust 实现，如果不可用则回退到 Python 实现。
    """
    _log_backend_once("merge_pair_all_words_with_deltas", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        return rust_merge_pair_all_words_with_deltas(words, left, right, merged)
    else:
        return python_merge_pair_all_words_with_deltas(words, left, right, merged)


def preprocess_and_pretokenize(
    text: str,
    special_tokens: list[str],
) -> list[list[bytes]]:
    """预处理和预分词。

    优先使用 Rust 实现，如果不可用则回退到 Python 实现。
    """
    _log_backend_once("preprocess_and_pretokenize", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        return rust_preprocess_and_pretokenize(text, special_tokens)
    else:
        # Python 实现需要额外参数，这里简化调用
        return python_preprocess_and_pretokenize(
            text, special_tokens, metrics_callback=None, num_workers=1
        )


class WordsChunkWithIndex:
    """带倒排索引的 words chunk。

    优先使用 Rust 实现，如果不可用则回退到 Python 实现。
    """

    def __init__(self, words: list[list[bytes]]):
        _log_backend_once("WordsChunkWithIndex.__init__", rust=RUST_AVAILABLE)
        if RUST_AVAILABLE:
            self._impl = RustWordsChunkWithIndex(words)
            self._is_rust = True
        else:
            self._impl = PythonWordsChunkWithIndex(words)
            self._is_rust = False

    def build_index(self) -> None:
        """构建倒排索引。"""
        self._impl.build_index()

    @property
    def _index_built(self) -> bool:
        """与 `_merge_optimizer.count_pairs_with_index` 兼容：Rust 侧在 merge 时会按需建索引。"""
        if self._is_rust:
            return True
        return self._impl._index_built

    def merge_pair_with_deltas(
        self, left: bytes, right: bytes, merged: bytes
    ) -> dict[tuple[bytes, bytes], int]:
        """使用倒排索引加速 merge。"""
        return self._impl.merge_pair_with_deltas(left, right, merged)

    @property
    def words(self) -> list[list[bytes]]:
        """获取 words。"""
        if self._is_rust:
            return self._impl.get_words()
        else:
            return self._impl.words

    def save(self, path) -> None:
        """序列化到磁盘。"""
        if self._is_rust:
            # Rust 实现暂不支持序列化，回退到 Python
            # 将 Rust words 转换为 Python 格式后保存
            from pathlib import Path
            import pickle

            payload = {
                "words": self.words,
                "pair_index": {},  # Rust 索引不导出
                "index_built": True,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self._impl.save(path)

    @classmethod
    def load(cls, path):
        """从磁盘加载。"""
        if RUST_AVAILABLE:
            # 加载后重新构建索引
            import pickle

            with open(path, "rb") as f:
                payload = pickle.load(f)

            chunk = cls(payload["words"])
            if payload.get("index_built"):
                chunk.build_index()
            return chunk
        else:
            return PythonWordsChunkWithIndex.load(path)


# 导出接口
__all__ = [
    "WordsChunkWithIndex",
    "count_pairs",
    "merge_pair_all_words_with_deltas",
    "preprocess_and_pretokenize",
    "RUST_AVAILABLE",
]
