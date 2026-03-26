"""BPE 编码/解码：与 GPT-2 / tiktoken 行为对齐（给定相同 vocab + merges）。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from llm_from_scratch.bpe_tokenizer._pat import ENCODE_SPLIT_PATTERN


class BPETokenizer:
    """由 vocab、merges 构造，支持 encode / decode / encode_iterable。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.special_tokens: list[str] = list(special_tokens or [])
        self._special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.bytes_to_id: dict[bytes, int] = {}
        for i, b in self.vocab.items():
            self.bytes_to_id[b] = i
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            (a, b): idx for idx, (a, b) in enumerate(merges)
        }
        self._pat = ENCODE_SPLIT_PATTERN

    def _bpe_encode_piece(self, piece: str) -> list[int]:
        if not piece:
            return []
        parts: list[bytes] = [bytes([x]) for x in piece.encode("utf-8")]
        while len(parts) > 1:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                r = self.merge_ranks.get(pair)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2 :]
        return [self.bytes_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched: str | None = None
            for st in self._special_sorted:
                if text.startswith(st, i):
                    matched = st
                    break
            if matched is not None:
                key = matched.encode("utf-8")
                out.append(self.bytes_to_id[key])
                i += len(matched)
                continue
            next_sp = n
            for st in self._special_sorted:
                j = text.find(st, i)
                if j != -1 and j < next_sp:
                    next_sp = j
            segment = text[i:next_sp]
            for frag in self._pat.findall(segment):
                out.extend(self._bpe_encode_piece(frag))
            i = next_sp
        return out

    def decode(self, token_ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in token_ids)
        return raw.decode("utf-8", errors="replace")

    def encode_iterable(self, chunks: Iterable[str]) -> Iterator[int]:
        """
        流式编码：在块边界上缓冲，按 ENCODE_SPLIT_PATTERN 从缓冲区前缀切出完整片段再 BPE。
        特殊串若可能被截断，会等待后续块。
        """
        buf = ""
        it = iter(chunks)
        max_sp = max((len(s) for s in self._special_sorted), default=0)

        def _need_more_for_special(prefix: str) -> bool:
            if not prefix:
                return False
            return any(
                st.startswith(prefix) and len(st) > len(prefix) for st in self._special_sorted
            )

        while True:
            try:
                buf += next(it)
            except StopIteration:
                break
            while buf:
                consumed = False
                for st in self._special_sorted:
                    if buf.startswith(st):
                        yield self.bytes_to_id[st.encode("utf-8")]
                        buf = buf[len(st) :]
                        consumed = True
                        break
                if consumed:
                    continue
                if _need_more_for_special(buf[:max_sp]) and len(buf) < max_sp:
                    break
                m = self._pat.match(buf)
                if m:
                    frag = m.group(0)
                    buf = buf[m.end() :]
                    for tid in self._bpe_encode_piece(frag):
                        yield tid
                    continue
                if _need_more_for_special(buf):
                    break
                break

        if buf:
            yield from self.encode(buf)


def make_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    return BPETokenizer(vocab, merges, special_tokens)


__all__ = ["BPETokenizer", "make_tokenizer"]
