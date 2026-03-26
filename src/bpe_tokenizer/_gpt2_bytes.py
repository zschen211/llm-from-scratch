"""GPT-2 字节表顺序与 bytes↔unicode 展示映射（与 tests.common 一致）。"""

from functools import lru_cache


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """字节值 0..255 -> GPT-2 用于展示/词表键的 unicode 单字符。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def gpt2_byte_positions() -> list[int]:
    """返回 256 个字节值在 GPT-2 词表中的顺序（对应参考 vocab 中 id 1..256）。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
    assert len(bs) == 256
    return bs
