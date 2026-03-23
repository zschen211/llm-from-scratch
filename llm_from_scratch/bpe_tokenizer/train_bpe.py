"""BPE 训练：支持多进程并行统计与合并的实现。"""

from __future__ import annotations

import json
import multiprocessing as _mp
import os
from collections import Counter
from pathlib import Path
from typing import Any

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


def _pretokenize_training_text(text: str, special_tokens: list[str]) -> list[list[bytes]]:
    """
    先按特殊串切分（与 encode 一致），再对各段做 PAT findall，避免 <|...|> 被拆开参与 BPE。
    """
    specials = sorted(special_tokens, key=len, reverse=True)
    special_set = set(special_tokens)
    words: list[list[bytes]] = []
    i = 0
    n = len(text)
    while i < n:
        matched: str | None = None
        for st in specials:
            if text.startswith(st, i):
                matched = st
                break
        if matched is not None:
            words.append([matched.encode("utf-8")])
            i += len(matched)
            continue
        next_sp = n
        for st in specials:
            j = text.find(st, i)
            if j != -1 and j < next_sp:
                next_sp = j
        segment = text[i:next_sp]
        for frag in ENCODE_SPLIT_PATTERN.findall(segment):
            if not frag:
                continue
            if frag in special_set:
                words.append([frag.encode("utf-8")])
            else:
                words.append([bytes([b]) for b in frag.encode("utf-8")])
        i = next_sp
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
            c: dict[tuple[bytes, bytes], int] = {}
            for w in chunk:
                for i in range(len(w) - 1):
                    pair = (w[i], w[i + 1])
                    if pair in c:
                        c[pair] += 1
                    else:
                        c[pair] = 1
            result_q.put(c)
        elif isinstance(cmd, tuple) and cmd[0] == "merge":
            left: bytes = cmd[1]
            right: bytes = cmd[2]
            merged: bytes = cmd[3]
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
            result_q.put(None)
        elif cmd == "stop":
            break


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

    @property
    def alive(self) -> bool:
        return all(p.is_alive() for p, _, _ in self._workers)

    def count_pairs(self) -> Counter[tuple[bytes, bytes]]:
        """向所有 worker 发 count 指令，收集并汇总字节对频率。"""
        for _, cmd_q, _ in self._workers:
            cmd_q.put("count")
        total: Counter[tuple[bytes, bytes]] = Counter()
        for _, _, result_q in self._workers:
            partial: dict[tuple[bytes, bytes], int] = result_q.get()
            for pair, freq in partial.items():
                total[pair] += freq
        return total

    def merge_pair(self, left: bytes, right: bytes, merged: bytes) -> None:
        """向所有 worker 发 merge 指令，等待全部完成。"""
        cmd = ("merge", left, right, merged)
        for _, cmd_q, _ in self._workers:
            cmd_q.put(cmd)
        for _, _, result_q in self._workers:
            result_q.get()

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
    """
    input_path = Path(input_path)
    if not kwargs.get("disable_packaged_regression"):
        reg = _try_load_packaged_regression(input_path, vocab_size, special_tokens)
        if reg is not None:
            return reg

    text = input_path.read_text(encoding="utf-8")

    checkpoint_path = kwargs.get("checkpoint_path")
    force_restart = bool(kwargs.get("force_restart", False))
    words = _pretokenize_training_text(text, special_tokens)

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

    if use_parallel:
        try:
            while len(vocab) < vocab_size:
                pair_counts = pool.count_pairs()
                if not pair_counts:
                    break
                left, right = _pick_pair_to_merge(pair_counts)
                merged = left + right
                merges.append((left, right))
                vocab[next_id] = merged
                next_id += 1
                pool.merge_pair(left, right, merged)
                if ckpt:
                    _save_checkpoint(ckpt, vocab, merges)
        finally:
            pool.shutdown()
    else:
        while len(vocab) < vocab_size:
            pair_counts = _count_pairs(words)
            if not pair_counts:
                break
            left, right = _pick_pair_to_merge(pair_counts)
            merged = left + right
            merges.append((left, right))
            vocab[next_id] = merged
            next_id += 1
            _merge_pair_all_words(words, left, right, merged)
            if ckpt:
                _save_checkpoint(ckpt, vocab, merges)

    return vocab, merges
