"""BPE merge 优化：倒排索引 + 磁盘 offload。

核心优化：
1. 倒排索引：pair → set[word_idx]，只处理包含目标 pair 的 words
2. chunk-local 索引：每个 chunk 维护自己的局部索引
3. 磁盘 offload：索引随 chunk 一起序列化，按需加载/卸载
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


class WordsChunkWithIndex:
    """带倒排索引的 words chunk，支持磁盘序列化。"""

    __slots__ = ("words", "pair_index", "_index_built")

    def __init__(self, words: list[list[bytes]]):
        self.words = words
        self.pair_index: dict[tuple[bytes, bytes], set[int]] = {}
        self._index_built = False

    def build_index(self) -> None:
        """构建倒排索引：pair → set[word_idx]。"""
        if self._index_built:
            return

        self.pair_index = defaultdict(set)
        for word_idx, word in enumerate(self.words):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_index[pair].add(word_idx)

        # 转为普通 dict 以节省内存
        self.pair_index = dict(self.pair_index)
        self._index_built = True

    def merge_pair_with_deltas(
        self, left: bytes, right: bytes, merged: bytes
    ) -> dict[tuple[bytes, bytes], int]:
        """使用倒排索引加速 merge，只处理包含目标 pair 的 words。

        返回 pair 频率的增量变化。
        """
        if not self._index_built:
            self.build_index()

        target_pair = (left, right)
        affected_word_indices = self.pair_index.get(target_pair, set())

        if not affected_word_indices:
            return {}

        delta = defaultdict(int)

        # 需要更新索引的 pairs
        pairs_to_remove_from_index: set[tuple[bytes, bytes]] = set()
        pairs_to_add_to_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

        for word_idx in affected_word_indices:
            word = self.words[word_idx]
            word_len = len(word)

            if word_len < 2:
                continue

            # 记录旧的 pairs（用于更新索引）
            old_pairs_in_word: set[tuple[bytes, bytes]] = set()
            for i in range(word_len - 1):
                old_pairs_in_word.add((word[i], word[i + 1]))

            # 执行 merge
            out: list[bytes] = []
            removed_pair_indices: set[int] = set()
            created_merge_positions: list[int] = []

            k = 0
            did_merge = False

            while k < word_len:
                if k + 1 < word_len and word[k] == left and word[k + 1] == right:
                    did_merge = True

                    # 记录被移除的 pair 索引
                    if k > 0:
                        removed_pair_indices.add(k - 1)
                    removed_pair_indices.add(k)
                    if k + 2 < word_len:
                        removed_pair_indices.add(k + 1)

                    created_merge_positions.append(len(out))
                    out.append(merged)
                    k += 2
                else:
                    out.append(word[k])
                    k += 1

            if not did_merge:
                continue

            # 计算 delta
            for i in removed_pair_indices:
                delta[(word[i], word[i + 1])] -= 1

            out_len = len(out)
            for pos in created_merge_positions:
                if pos > 0:
                    delta[(out[pos - 1], out[pos])] += 1
                if pos < out_len - 1:
                    delta[(out[pos], out[pos + 1])] += 1

            # 更新 words
            self.words[word_idx] = out

            # 记录新的 pairs（用于更新索引）
            new_pairs_in_word: set[tuple[bytes, bytes]] = set()
            for i in range(len(out) - 1):
                new_pairs_in_word.add((out[i], out[i + 1]))

            # 计算索引更新
            removed_pairs = old_pairs_in_word - new_pairs_in_word
            added_pairs = new_pairs_in_word - old_pairs_in_word

            for pair in removed_pairs:
                pairs_to_remove_from_index.add(pair)

            for pair in added_pairs:
                pairs_to_add_to_index[pair].add(word_idx)

        # 更新倒排索引
        for pair in pairs_to_remove_from_index:
            if pair in self.pair_index:
                self.pair_index[pair] -= affected_word_indices
                if not self.pair_index[pair]:
                    del self.pair_index[pair]

        for pair, word_indices in pairs_to_add_to_index.items():
            if pair not in self.pair_index:
                self.pair_index[pair] = set()
            self.pair_index[pair] |= word_indices

        return {p: v for p, v in delta.items() if v != 0}

    def save(self, path: Path) -> None:
        """序列化到磁盘（包含 words 和索引）。"""
        payload = {
            "words": self.words,
            "pair_index": {
                # 将 set 转为 list 以便序列化
                pair: list(indices) for pair, indices in self.pair_index.items()
            },
            "index_built": self._index_built,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> WordsChunkWithIndex:
        """从磁盘加载。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        chunk = cls(payload["words"])
        chunk.pair_index = {
            # 将 list 转回 set
            pair: set(indices) for pair, indices in payload["pair_index"].items()
        }
        chunk._index_built = payload["index_built"]
        return chunk

    def get_memory_estimate_mb(self) -> float:
        """估算内存占用（MB）。"""
        import sys

        words_size = sys.getsizeof(self.words)
        for word in self.words:
            words_size += sys.getsizeof(word)
            for token in word:
                words_size += sys.getsizeof(token)

        index_size = sys.getsizeof(self.pair_index)
        for pair, indices in self.pair_index.items():
            index_size += sys.getsizeof(pair) + sys.getsizeof(indices)

        return (words_size + index_size) / (1024 * 1024)


def count_pairs_with_index(
    chunk: WordsChunkWithIndex,
) -> dict[tuple[bytes, bytes], int]:
    """使用倒排索引统计 pair 频率。

    注意：索引只记录 pair 出现在哪些 words 中，不记录出现次数。
    因此仍需遍历 words 统计实际频率（但可以跳过不包含任何 pair 的 words）。
    """
    if not chunk._index_built:
        chunk.build_index()

    # 直接遍历所有 words 统计（索引用于加速 merge，不用于加速 count）
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for word in chunk.words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts

