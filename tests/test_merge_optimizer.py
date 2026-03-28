"""测试倒排索引优化的正确性。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_from_scratch.bpe_tokenizer._merge_optimizer import WordsChunkWithIndex


def test_inverted_index_basic():
    """测试基本的倒排索引构建和查询。"""
    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
        [b"h", b"e", b"l", b"p"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # 检查索引是否正确构建
    assert (b"h", b"e") in chunk.pair_index
    assert (b"l", b"l") in chunk.pair_index
    assert (b"o", b"r") in chunk.pair_index

    # 检查 word_idx 是否正确
    assert 0 in chunk.pair_index[(b"h", b"e")]  # "hello"
    assert 2 in chunk.pair_index[(b"h", b"e")]  # "help"
    assert 0 in chunk.pair_index[(b"l", b"l")]  # "hello"
    assert 1 in chunk.pair_index[(b"o", b"r")]  # "world"


def test_merge_with_index():
    """测试使用倒排索引的 merge 操作。"""
    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
        [b"h", b"e", b"l", b"p"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # Merge (b"l", b"l") -> b"ll"
    delta = chunk.merge_pair_with_deltas(b"l", b"l", b"ll")

    # 检查 merge 结果
    assert chunk.words[0] == [b"h", b"e", b"ll", b"o"]
    assert chunk.words[1] == [b"w", b"o", b"r", b"l", b"d"]  # 未受影响
    assert chunk.words[2] == [b"h", b"e", b"l", b"p"]  # 未受影响

    # 检查 delta
    assert delta[(b"l", b"l")] < 0  # (l, l) 被移除
    assert delta[(b"e", b"ll")] > 0  # (e, ll) 新增
    assert delta[(b"ll", b"o")] > 0  # (ll, o) 新增

    # 检查索引是否更新
    assert (b"l", b"l") not in chunk.pair_index  # 已被移除
    assert (b"e", b"ll") in chunk.pair_index
    assert (b"ll", b"o") in chunk.pair_index


def test_merge_no_affected_words():
    """测试 merge 不存在的 pair（快速返回）。"""
    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # Merge 不存在的 pair
    delta = chunk.merge_pair_with_deltas(b"x", b"y", b"xy")

    # 应该快速返回空 delta
    assert delta == {}

    # words 应该未被修改
    assert chunk.words[0] == [b"h", b"e", b"l", b"l", b"o"]
    assert chunk.words[1] == [b"w", b"o", b"r", b"l", b"d"]


def test_save_and_load():
    """测试序列化和反序列化。"""
    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # 保存到临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chunk.pkl"
        chunk.save(path)

        # 加载
        loaded_chunk = WordsChunkWithIndex.load(path)

        # 检查 words 是否一致
        assert loaded_chunk.words == chunk.words

        # 检查索引是否一致
        assert loaded_chunk.pair_index == chunk.pair_index
        assert loaded_chunk._index_built == chunk._index_built


def test_multiple_merges():
    """测试连续多次 merge。"""
    words = [
        [b"a", b"b", b"c", b"d"],
        [b"a", b"b", b"e", b"f"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # 第一次 merge: (a, b) -> ab
    delta1 = chunk.merge_pair_with_deltas(b"a", b"b", b"ab")
    assert chunk.words[0] == [b"ab", b"c", b"d"]
    assert chunk.words[1] == [b"ab", b"e", b"f"]
    assert (b"a", b"b") not in chunk.pair_index
    assert (b"ab", b"c") in chunk.pair_index
    assert (b"ab", b"e") in chunk.pair_index

    # 第二次 merge: (ab, c) -> abc
    delta2 = chunk.merge_pair_with_deltas(b"ab", b"c", b"abc")
    assert chunk.words[0] == [b"abc", b"d"]
    assert chunk.words[1] == [b"ab", b"e", b"f"]  # 未受影响
    assert (b"ab", b"c") not in chunk.pair_index
    assert (b"abc", b"d") in chunk.pair_index


if __name__ == "__main__":
    test_inverted_index_basic()
    test_merge_with_index()
    test_merge_no_affected_words()
    test_save_and_load()
    test_multiple_merges()
    print("All tests passed!")
