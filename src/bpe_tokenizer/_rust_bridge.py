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
        dump_words_chunk_py as rust_dump_words_chunk,
        load_words_chunk_py as rust_load_words_chunk,
        merge_pair_all_words_with_deltas_py as rust_merge_pair_all_words_with_deltas,
        preprocess_and_pretokenize_py as rust_preprocess_and_pretokenize,
        pretokenize_with_pat_py as rust_pretokenize_with_pat,
        train_bpe_full_py as rust_train_bpe_full,
    )

    RUST_AVAILABLE = True
    RUST_PAT_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    RUST_PAT_AVAILABLE = False
    RustWordsChunkWithIndex = None
    rust_count_pairs = None
    rust_dump_words_chunk = None
    rust_load_words_chunk = None
    rust_merge_pair_all_words_with_deltas = None
    rust_preprocess_and_pretokenize = None
    rust_pretokenize_with_pat = None
    rust_train_bpe_full = None
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


def pretokenize_with_pat(
    text: str,
    special_tokens: list[str],
    use_tiktoken_pat: bool = True,
) -> list[list[bytes]]:
    """使用 fancy-regex 的预分词（支持环视断言）。

    优先使用 Rust 实现，如果不可用则回退到 Python 实现。

    Args:
        text: 输入文本
        special_tokens: 特殊 token 列表
        use_tiktoken_pat: 是否使用 tiktoken GPT-2 PAT（默认 True），否则使用 CS336 PAT

    Returns:
        预分词后的 words 列表
    """
    _log_backend_once("pretokenize_with_pat", rust=RUST_PAT_AVAILABLE)
    if RUST_PAT_AVAILABLE:
        return rust_pretokenize_with_pat(text, special_tokens, use_tiktoken_pat)
    else:
        # 回退到 Python 实现
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
    "dump_words_chunk",
    "load_words_chunk",
    "merge_pair_all_words_with_deltas",
    "preprocess_and_pretokenize",
    "pretokenize_with_pat",
    "train_bpe_full",
    "RUST_AVAILABLE",
    "RUST_PAT_AVAILABLE",
]


def dump_words_chunk(path: str, words: list[list[bytes]]) -> None:
    """保存 words chunk 到文件（bincode 格式）。

    优先使用 Rust 实现，如果不可用则回退到 Python pickle。
    """
    _log_backend_once("dump_words_chunk", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        rust_dump_words_chunk(path, words)
    else:
        # 回退到 pickle
        import pickle

        with open(path, "wb") as f:
            pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_words_chunk(path: str) -> list[list[bytes]]:
    """从文件加载 words chunk（bincode 格式）。

    优先使用 Rust 实现，如果不可用则回退到 Python pickle。
    """
    _log_backend_once("load_words_chunk", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        return rust_load_words_chunk(path)
    else:
        # 回退到 pickle
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def train_bpe_full(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 4,
    stream_chunk_chars: int = 1_000_000,
    chunks_dir: str = ".bpe_chunks",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """完整的 BPE 训练流程（Rust 实现）。

    Args:
        input_path: 输入语料文件路径
        vocab_size: 目标词表大小
        special_tokens: 特殊 token 列表
        num_workers: 并行 worker 数量
        stream_chunk_chars: 流式读取的 chunk 大小（字符数）
        chunks_dir: chunk 文件存储目录

    Returns:
        (vocab, merges) 元组
    """
    _log_backend_once("train_bpe_full", rust=RUST_AVAILABLE)
    if RUST_AVAILABLE:
        return rust_train_bpe_full(
            input_path,
            vocab_size,
            special_tokens,
            num_workers,
            stream_chunk_chars,
            chunks_dir,
        )
    else:
        raise NotImplementedError("train_bpe_full 需要 Rust 实现，请安装 bpe_core")
