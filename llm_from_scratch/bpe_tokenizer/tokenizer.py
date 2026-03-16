import json
import os
import regex
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

# token_ids_v?.json 所在目录：项目根目录下的 tokenizer 文件夹
_TOKENIZER_DIR = Path(__file__).resolve().parent.parent.parent / "tokenizer"


class TextFileDataLoader:
    """按固定大小从文本文件中读取数据，并支持记录/恢复游标。"""

    def __init__(self, file_path: str | os.PathLike[str]):
        self._path = Path(file_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"训练文本文件不存在: {self._path}")
        self._cursor: int = 0  # 当前字节偏移

    def get_cursor(self) -> int:
        """返回当前文档游标（字节偏移）。"""
        return self._cursor

    def set_cursor(self, position: int) -> None:
        """设置文档游标位置（字节偏移）。"""
        if position < 0:
            raise ValueError("游标不能为负数")
        self._cursor = position

    def read_chunk(self, size: int) -> str:
        """
        从当前游标起按固定大小（字节数）读取一段文本，并推进游标。

        :param size: 要读取的字节数。
        :return: 解码后的文本；若已到文件末尾则可能短于 size 对应长度。
        """
        if size <= 0:
            raise ValueError("size 必须为正整数")
        with open(self._path, "rb") as f:
            f.seek(self._cursor)
            raw = f.read(size)
        self._cursor += len(raw)
        return raw.decode("utf-8", errors="replace")

    def reset_cursor(self) -> None:
        """将游标重置到文件开头。"""
        self._cursor = 0

    def __len__(self) -> int:
        """文件总字节数。"""
        return self._path.stat().st_size



class BPETokenizer:
    def __init__(
        self,
        vocab_version: int = 1,
        force_rebuild: bool = False,
    ):
        """
        :param vocab_version: 使用或创建 token_ids_v{version}.json 的版本号。
        :param force_rebuild: 为 True 时忽略已有文件，用默认 257 个符号重建 vocab 并写回文件。
        """
        self.vocab_version = vocab_version
        self.GPT2_SPLIT_PATTERN = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        self.WORD_END_TOKEN_ID = 256

        if force_rebuild:
            self.vocab = self._default_byte_vocab()
            self._save_vocab(vocab_version)
        else:
            path = self._token_ids_path(vocab_version)
            if path.is_file():
                self.vocab = self._load_vocab(vocab_version)
            else:
                self.vocab = self._default_byte_vocab()
                self._save_vocab(vocab_version)

    # def encode(self, text: str) -> list[int]:
    #     return [self.vocab[token] for token in text.split()]

    # def decode(self, ids: list[int]) -> str:
    #     return "".join([self.vocab[id] for id in ids])

    def pretokenize(self, text_chunk: str) -> list[list[int]]:
        fragments = self.GPT2_SPLIT_PATTERN.findall(text_chunk)
        byte_ids = [
            [b for b in frag.encode("utf-8")] + [self.WORD_END_TOKEN_ID]
            for frag in fragments
            if frag  # 过滤空字符串
        ]
        return byte_ids

    def _default_byte_vocab(self) -> dict[int, bytes]:
        id_to_tokens = {i: bytes([i]) for i in range(256)}
        id_to_tokens[256] = b"</w>"  # WORD_END 对应符号
        return id_to_tokens

    def _save_vocab(self, version: int) -> None:
        """将当前 vocab 持久化到 token_ids_v{version}.json。"""
        path = self._token_ids_path(version)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab_to_json_serializable(self.vocab), f, ensure_ascii=False)

    def _load_vocab(self, version: int) -> dict[int, bytes]:
        """从 token_ids_v{version}.json 加载 vocab。"""
        path = self._token_ids_path(version)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self._vocab_from_json_serializable(data)

    def _vocab_to_json_serializable(self, vocab: dict[int, bytes]) -> dict[str, list[int]]:
        """vocab id -> bytes 转为可 JSON 序列化的 id -> 字节值列表。"""
        return {str(k): list(v) for k, v in vocab.items()}


    def _vocab_from_json_serializable(self, data: dict[str, list[int]]) -> dict[int, bytes]:
        """从 JSON 反序列化恢复 vocab。"""
        return {int(k): bytes(v) for k, v in data.items()}


    def _token_ids_path(self, version: int) -> Path:
        """指定版本的 token_ids 文件路径。"""
        return _TOKENIZER_DIR / f"token_ids_v{version}.json"


class BPETokenizerTrainer:
    """整合训练文本文件读取与多进程管理，用于 BPE 训练流程的并行化。"""

    def __init__(
        self,
        file_path: str | os.PathLike[str],
        vocab_version: int = 1,
        force_rebuild: bool = False,
        n_processes: int = 1,
    ):
        """
        :param file_path: 训练文本文件路径，供 DataLoader 读取。
        :param vocab_version: 传给 BPETokenizer 的 vocab 版本号。
        :param force_rebuild: 传给 BPETokenizer，是否强制重建 vocab。
        :param n_processes: 进程池大小。>1 时创建进程池；<=1 时不创建池。
        """
        self._loader = TextFileDataLoader(file_path)
        self._tokenizer = BPETokenizer(
            vocab_version=vocab_version,
            force_rebuild=force_rebuild,
        )
        self._n_processes = max(0, int(n_processes))
        self._pool: Pool | None = None
        if self._n_processes > 1:
            self._pool = Pool(processes=self._n_processes)

    @property
    def tokenizer(self) -> BPETokenizer:
        """内部使用的 BPETokenizer。"""
        return self._tokenizer

    @property
    def loader(self) -> TextFileDataLoader:
        """文件数据加载器，用于按块读取训练文本。"""
        return self._loader

    @property
    def pool(self) -> Optional[Pool]:
        """进程池，仅当 n_processes > 1 时非 None。"""
        return self._pool

    @property
    def n_processes(self) -> int:
        """初始化时配置的进程数。"""
        return self._n_processes

    def read_chunk(self, size: int) -> str:
        """从训练文件中读取一段文本（委托给 loader）。"""
        return self._loader.read_chunk(size)

    def get_cursor(self) -> int:
        """当前文档游标（字节偏移）。"""
        return self._loader.get_cursor()

    def set_cursor(self, position: int) -> None:
        """设置文档游标位置。"""
        self._loader.set_cursor(position)

    def reset_cursor(self) -> None:
        """将游标重置到文件开头。"""
        self._loader.reset_cursor()

    def map_parallel(self, func: Any, iterable: Any, chunksize: int = 1) -> Any:
        """
        使用进程池并行 map。仅当 n_processes > 1 时真正并行，否则退化为顺序 map。
        :param func: 可 pickle 的调用对象（如顶层函数）。
        :param iterable: 可迭代参数。
        :param chunksize: 每个任务块大小，仅多进程时有效。
        """
        if self._pool is not None:
            return self._pool.imap(func, iterable, chunksize=chunksize)
        return map(func, iterable)

    def close_pool(self) -> None:
        """关闭进程池，释放资源。使用完毕后建议调用。"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "BPETokenizerTrainer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_pool()

    def __len__(self) -> int:
        """训练文件总字节数（委托给 loader）。"""
        return len(self._loader)


def _find_tinystories_in_data() -> Path:
    """在 data 目录下查找 TinyStories 相关文件。"""
    data_dir = Path("data")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")
    candidates = list(data_dir.glob("*[Tt]iny*[Ss]tories*.txt"))
    if not candidates:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 TinyStories 文本文件")
    return candidates[0]


def main() -> None:
    tinystories_path = _find_tinystories_in_data()
    print(f"使用文件: {tinystories_path}")

    with BPETokenizerTrainer(tinystories_path, n_processes=1) as trainer:
        tokenizer = trainer.tokenizer
        chunk_size = 8192  # 每次读 8KB
        chunk = trainer.read_chunk(chunk_size)
        fragments = tokenizer.pretokenize(chunk)

        print(f"读取 {len(chunk)} 个字符，pretokenize 得到 {len(fragments)} 个片段")
        print("前 20 个片段(原始):", fragments[:20])
        print("vocab 长度:", len(tokenizer.vocab))


if __name__ == "__main__":
    main()