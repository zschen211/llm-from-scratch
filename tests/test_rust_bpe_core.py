"""测试 Rust 实现的正确性和性能。"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("bpe_core")

from llm_from_scratch.bpe_tokenizer._rust_bridge import RUST_AVAILABLE

assert RUST_AVAILABLE is True


def test_count_pairs():
    """测试 count_pairs 函数。"""
    from llm_from_scratch.bpe_tokenizer._rust_bridge import count_pairs

    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
        [b"h", b"e", b"l", b"p"],
    ]

    result = count_pairs(words, min_freq=1)

    assert (b"h", b"e") in result
    assert result[(b"h", b"e")] == 2  # "hello" 和 "help"
    assert (b"l", b"l") in result
    assert result[(b"l", b"l")] == 1  # "hello"

    print("✓ count_pairs test passed")


def test_merge_pair():
    """测试 merge_pair_all_words_with_deltas 函数。"""
    from llm_from_scratch.bpe_tokenizer._rust_bridge import (
        merge_pair_all_words_with_deltas,
    )

    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
        [b"h", b"e", b"l", b"p"],
    ]

    delta = merge_pair_all_words_with_deltas(words, b"l", b"l", b"ll")

    # 检查 merge 结果
    assert words[0] == [b"h", b"e", b"ll", b"o"]
    assert words[1] == [b"w", b"o", b"r", b"l", b"d"]  # 未受影响
    assert words[2] == [b"h", b"e", b"l", b"p"]  # 未受影响

    # 检查 delta
    assert (b"l", b"l") in delta
    assert delta[(b"l", b"l")] < 0  # 被移除

    print("✓ merge_pair_all_words_with_deltas test passed")


def test_words_chunk_with_index():
    """测试 WordsChunkWithIndex 类。"""
    from llm_from_scratch.bpe_tokenizer._rust_bridge import WordsChunkWithIndex

    words = [
        [b"h", b"e", b"l", b"l", b"o"],
        [b"w", b"o", b"r", b"l", b"d"],
        [b"h", b"e", b"l", b"p"],
    ]

    chunk = WordsChunkWithIndex(words)
    chunk.build_index()

    # 测试 merge
    delta = chunk.merge_pair_with_deltas(b"l", b"l", b"ll")

    # 检查结果
    result_words = chunk.words
    assert result_words[0] == [b"h", b"e", b"ll", b"o"]

    print("✓ WordsChunkWithIndex test passed")


def benchmark_count_pairs():
    """性能基准测试：count_pairs。"""
    from llm_from_scratch.bpe_tokenizer._rust_bridge import count_pairs

    # 生成测试数据
    words = []
    for i in range(10000):
        word = [bytes([j % 256]) for j in range(i % 20 + 1)]
        words.append(word)

    # 测试性能
    t0 = time.perf_counter()
    result = count_pairs(words, min_freq=1)
    elapsed = time.perf_counter() - t0

    print(f"✓ count_pairs benchmark: {len(result)} pairs in {elapsed*1000:.2f}ms")


def benchmark_merge_pair():
    """性能基准测试：merge_pair_all_words_with_deltas。"""
    from llm_from_scratch.bpe_tokenizer._rust_bridge import (
        merge_pair_all_words_with_deltas,
    )

    # 生成测试数据
    words = []
    for i in range(10000):
        word = [b"a", b"b"] * 10
        words.append(word)

    # 测试性能
    t0 = time.perf_counter()
    delta = merge_pair_all_words_with_deltas(words, b"a", b"b", b"ab")
    elapsed = time.perf_counter() - t0

    print(
        f"✓ merge_pair_all_words_with_deltas benchmark: {len(delta)} deltas in {elapsed*1000:.2f}ms"
    )


if __name__ == "__main__":
    print("=== Testing Rust BPE Core Module ===\n")

    try:
        test_count_pairs()
        test_merge_pair()
        test_words_chunk_with_index()
        print()
        benchmark_count_pairs()
        benchmark_merge_pair()
        print("\n=== All tests passed! ===")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
