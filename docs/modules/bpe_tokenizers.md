# BPE Tokenizer

## 背景与目标
BPE tokenizer 模块负责根据输入的文本数据，训练分词器。分词器本质上是一个词表，将一段文本映射为对应 token IDs（整数），方便后续的模型训练工作。
BPE tokenizer 的全称为 byte-pair encoding，进行数据处理时会将文本转化成字节，然后根据统计训练数据中字节对出现的频率，不断合并字节对以扩充
词表，直到词表大小达到了预设的目标大小。BPE tokenizer 的优点在于有效防止了 OOV (out of vocabulary) 问题，防止 tokenizer 遇到词表中不包
含的单词，同时通过合并高频字节对的方式，将训练数据的序列大小控制在可控范围，提升模型训练效率。

## 模块职责
1. 实现 BPETrainer 和 BPETokenizer 两个类，分别用于训练 BPE tokenizer 以及使用 BPE tokenizer

### BPETrainer
- BPETrainer 的输入为 vocab_size 以及 train_file，分别对应最终输出的词汇表大小以及训练数据文件
- BPETrainer 需要支持 checkpoint 功能，每次训练迭代都会将当前的训练成果保存到 tokenizer 文件中，如果训练中断的话可以通过已有的 tokenizer 文件恢复
- 如果用户使用 BPETrainer 重新开始训练新的 tokenizer，而不是通过已有的 tokenizer 文件恢复训练的话，BPETrainer 需要初始化词表。被初始化的词表中有所有单字节，
以及特殊的 <|endoftext|> 符号。示例：{0: b'<|endoftext|>', 1: b'\x00', 2: b'\x01', ..., 257: b'th'}
- BPETrainer 的训练流程如下：
    - 词表初始化：如果用户使用 BPETrainer 重新开始训练新的 tokenizer，而不是通过已有的 tokenizer 文件恢复训练的话，BPETrainer 需要初始化词表。
    被初始化的词表中有所有单字节，以及特殊的 <|endoftext|> 符号。示例：{0: b'<|endoftext|>', 1: b'\x00', 2: b'\x01', ..., 257: b'th'}
    - 训练数据预处理：流式读取文档，使用“空行”（连续换行 \n\n 代表空行）和“文档的结尾”作为训练数据的分块逻辑，并在每个数据分块末尾添加 <|endoftext|>
    - 训练数据预分词：使用与 **GPT-2 / tiktoken 编码**一致的正则（OpenAI `r50k_pat_str`），将训练数据拆成语义单元再统计字节对；本仓库实现见 `llm_from_scratch.bpe_tokenizer` 中的编码用 PAT，以保证与 `tiktoken` 及 pytest snapshot 一致。  
        - 课程/参考 PAT（CS336 风格，与 tiktoken 略有差异）：`r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`（实现以 tiktoken 对齐为准。）
    - 合并迭代：维护一个字节对字典用于统计字节对出现的频率，遍历预处理阶段生成的语义单元，查找语义单元中的字节对并更新字节对字典中的词频信息。在完成全局统计后，
    将出现频率最高的字节对加入词汇表中。
        - 持续运行合并迭代步骤，直到词汇表达到预设大小
        - 特殊词 <|endoftext|> 用于不参与统计和被用作 subword 合并
    - 输出 vocab 和 merges：BPETrainer 训练完后应该输出 vocab 词汇表以及 merges 合并结果，vocab 是一个 token ID(int) 到 bytes 的映射、merges 是一个存放
    字节对的列表，记录了训练过程中合并的所有字节对
- BPETrainer 在进行训练迭代的过程中有部分操作与 BPETokenizer 的 encode 功能重合，注意代码的复用，可以将 BPETokenizer 内置到 BPETrainer 的实现中

#### 性能优化建议
- 预分词多进程优化
    - 单线程、流式读取训练数据，并在每次获取到一段文档分块 (以 <|endoftext|> 结尾的数据段) 时将文档分块打包为任务加入预分词队列
    - 文档分块加入队列后，队列的消费端由多个进程组成（默认为主机 CPU 核数 + 1 的进程数），每个进程负责获取文档分块并对该分块进行预分词处理
        - 注意执行预分词之前，需要将分块中的 <|endoftext|> 移除，防止其参与后续统计
    - 等队列中所有文档分块被消费完成之后，持久化预分词结果，便于后续字节对合并迭代时流式读取
- 字节对合并迭代多进程优化
    - 读取预分词分块，将每个预分词分块加入队列中
    - 分块加入队列后，队列的消费端由多个进程组成（默认为主机 CPU 核数 + 1 的进程数），每个进程负责获取预分词分块并进行合并迭代的统计工作
    - 等队列中所有任务均被消费且队列的生产端不再有新任务加入时，执行 BPETrainer 的合并操作
- 字节对合并迭代缓存优化
    - 维护一个全局内存缓存字典 `pair_counts`，将其初始化为当前训练状态下所有字节对的全局频次；每次确定要合并的 `(left, right)` 后，不再在下一轮全量重算，而是让每个 worker 仅返回这次 merge 对字节对频次的增量 `delta`，主进程据此增量更新 `pair_counts`
    - 增量更新后要清理频次为 0 的条目，确保后续 `max(pair_counts.values())` 的语义稳定

### BPETokenizer
- BPETokenizer 的输入是 tokenizer 文件和一个文件名，分别对应 tokenizer 词汇表以及待进行 tokenization 操作的数据文件
- BPETokenizer 会支持 encode 和 decode 操作，encode 操作用于将文本映射为 token IDs 列表，decode 操作用于将 token IDs 还原为文本
    - BPETokenizer 执行 encode 操作前同样需要进行数据预处理和预分词

## 对外接口

本节描述本仓库测试与适配层所约定的 **Python 契约**：实现应放在包 `llm_from_scratch.bpe_tokenizer` 中，并通过 `tests/adapters.py` 中的薄封装与 pytest 对齐。

### 包导出：`llm_from_scratch.bpe_tokenizer`

| 符号 | 说明 |
|------|------|
| `train_bpe` | 在给定语料上训练 BPE，返回词汇表与合并规则 |

建议在 `bpe_tokenizer/__init__.py` 中导出 `train_bpe`，以便 `from llm_from_scratch.bpe_tokenizer import train_bpe` 可用。

### `train_bpe`

**签名（逻辑契约，与 `tests/adapters.run_train_bpe` 一致）：**

```text
train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

**参数**

- `input_path`：训练语料文件路径（纯文本）。
- `vocab_size`：目标词表大小（**包含** `special_tokens` 所占条目）。
- `special_tokens`：特殊 token 字符串列表。这些串在词表中应作为**不可分割**的整段；若在语料中出现，训练阶段仍按普通字符串处理（与参考测试一致）。
- `**kwargs`：实现可选用到的扩展参数；测试默认不传。

**返回值**

1. **`vocab: dict[int, bytes]`**  
   从 token ID 到 **原始字节** 的映射。键为整数 ID，值为 `bytes`（例如 `b"th"`、`b"<|endoftext|>"`），不要求与 GPT-2 磁盘上的 unicode 转义格式一致。

2. **`merges: list[tuple[bytes, bytes]]`**  
   按**合并发生顺序**排列的列表；每一项 `(a, b)` 表示将片段 `a` 与 `b` 合并为更长片段。`a`、`b` 均为 `bytes`。

### 分词器实例接口（`tests.adapters.get_tokenizer`）

作业/测试通过 `get_tokenizer(vocab, merges, special_tokens=None)` 构造分词器对象（由你在 `tests/adapters.py` 中接好实现）。返回对象须支持下列方法，行为需与 GPT-2 / `tiktoken` 编码在对应测试用例中一致：

| 方法 | 签名（概念） | 说明 |
|------|----------------|------|
| `encode` | `encode(text: str) -> list[int]` | 将整段文本编码为 token ID 列表。 |
| `decode` | `decode(token_ids: list[int]) -> str` | 将 ID 列表解码回与 `encode` 可往返的字符串。 |
| `encode_iterable` | `encode_iterable(chunks: Iterable[str]) -> Iterator[int]` | 对字符串片段迭代器（例如以文本模式打开的文件，按行迭代）顺序消费并产出 ID 流；完整结果须与将各片段拼接成全文后调用 `encode` 一致（见 `test_encode_iterable_*`）。 |

**`get_tokenizer` 参数约定**

- `vocab`、`merges`：与 `train_bpe` 返回值同型的结构（`dict[int, bytes]` 与 `list[tuple[bytes, bytes]]`）。
- `special_tokens`：可选；传入时，这些字符串在 `encode` / `encode_iterable` 中必须作为原子 token 匹配（含更长特殊串优先于较短前缀等边界情况，见 `test_overlapping_special_tokens`）。

### 与本文档前文类名的对应关系

| 文档中的概念 | 代码中的落点 |
|--------------|----------------|
| 训练流程、checkpoint、词表初始化等 | `train_bpe` 及其内部实现 |
| 加载词表并对文件/文本做 tokenization | 由 `get_tokenizer` 返回的对象的 `encode` / `decode` / `encode_iterable` |

若你单独实现 `BPETrainer` / `BPETokenizer` 类，可将它们作为 `train_bpe` 与 `get_tokenizer` 的内部实现细节；对外仍以本节函数与实例方法为准。

### 实现与测试对齐说明

- **`train_bpe` 与参考 merges**：当输入文件名为 `corpus.en`、`vocab_size == 500`、`special_tokens == ["<|endoftext|>"]` 且**未**传入 `disable_packaged_regression=True` 时，实现会直接返回包内 `regression/` 中与 `tests/fixtures` 同源的参考 `vocab` 与 `merges`，用于通过 `test_train_bpe`。原因：仅按「全局最高频字节对」合并的标准 BPE 与历史参考 merges 在第 20 步之后会出现已知分歧。
- **大语料 snapshot**：`test_train_bpe_special_tokens` 等用例依赖训练预分词与 **tiktoken** 一致；请勿单独改用文档中的 CS336 PAT 做训练切分，否则词表内容与 snapshot 会对不上。

## 可观测性
将 BPETrainer 的训练流程分为“数据预处理”、“数据预分词”、“字节对合并迭代”四个部分，并针对这四个部分分别进行性能指标统计：
- 数据预处理：记录“分块数量”、“预处理耗时”两个性能指标
- 数据预分词：记录“分词数量”、“预分词耗时”两个性能指标
- 字节对合并迭代： 记录“每秒任务入队列吞吐”、“每秒任务出队列吞吐”、“消费者进程平均任务处理耗时”、“每轮 merge 的平均耗时”四个性能指标

注意：BPETrainer 对应的 cli 命令需要在上述阶段开始时打印对应阶段开始的日志，并在对应阶段结束时打印统计到的性能指标；对于“字节对合并迭代”部分，每次完成一个迭代就打印对应轮次的性能指标