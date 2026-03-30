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
- BPETrainer 使用流式模式处理训练数据，支持大文件训练
- BPETrainer 的训练流程如下：
    - 词表初始化：初始化词表，包含所有单字节以及特殊的 <|endoftext|> 符号。示例：{0: b'<|endoftext|>', 1: b'\x00', 2: b'\x01', ..., 257: b'th'}
    - 训练数据预处理：流式读取文档，使用”空行”（连续换行 \n\n 代表空行）和”文档的结尾”作为训练数据的分块逻辑，并在每个数据分块末尾添加 <|endoftext|>
    - 训练数据预分词：使用与 **GPT-2 / tiktoken 编码**一致的正则（OpenAI `r50k_pat_str`），将训练数据拆成语义单元再统计字节对；本仓库实现见 `llm_from_scratch.bpe_tokenizer` 中的编码用 PAT，以保证与 `tiktoken` 及 pytest snapshot 一致。  
        - 课程/参考 PAT（CS336 风格，与 tiktoken 略有差异）：`r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`（实现以 tiktoken 对齐为准。）
    - 合并迭代：维护一个字节对字典用于统计字节对出现的频率，遍历预处理阶段生成的语义单元，查找语义单元中的字节对并更新字节对字典中的词频信息。在完成全局统计后，
    将出现频率最高的字节对加入词汇表中。
        - 在第一次迭代后建立倒排索引，之后迭代统计的过程都基于倒排索引进行统计，而不是反复遍历 chunk 文件；例如说，在统计了 pair_count {(e,w) = 3} 之后，同步建立索引 {(e,w) -> [new, newer, newest]}，在之后的合并迭代流程中根据倒排索引进行计数、合并决策
        - 持续运行合并迭代步骤，直到词汇表达到预设大小
        - 特殊词 <|endoftext|> 用于不参与统计和被用作 subword 合并
    - 输出 vocab 和 merges：BPETrainer 训练完后应该输出 vocab 词汇表以及 merges 合并结果，vocab 是一个 token ID(int) 到 bytes 的映射、merges 是一个存放
    字节对的列表，记录了训练过程中合并的所有字节对
- BPETrainer 在进行训练迭代的过程中有部分操作与 BPETokenizer 的 encode 功能重合，注意代码的复用，可以将 BPETokenizer 内置到 BPETrainer 的实现中

### BPETokenizer
- BPETokenizer 的输入是 tokenizer 文件和一个文件名，分别对应 tokenizer 词汇表以及待进行 tokenization 操作的数据文件
- BPETokenizer 支持 encode 和 decode 操作，encode 操作用于将文本映射为 token IDs 列表，decode 操作用于将 token IDs 还原为文本
    - BPETokenizer 执行 encode 操作前同样需要进行数据预处理和预分词

## 性能优化
- 流式模式
    - BPE 训练统一使用流式模式，支持大文件训练
    - 流式读取训练数据，按 `stream_chunk_chars` 字符分块
    - 预分词后累积到内存，达到 `stream_memory_target_percent` 阈值时落盘为 chunk 文件
    - 合并迭代时分批加载 chunk 文件，处理后回写
- 预分词多进程优化
    - 单线程、流式读取训练数据，并在每次获取到一段文档分块 (以 <|endoftext|> 结尾的数据段) 时将文档分块打包为任务加入预分词队列
    - 文档分块加入队列后，队列的消费端由多个进程组成（默认为主机 CPU 核数 + 1 的进程数），每个进程负责获取文档分块并对该分块进行预分词处理
        - 注意执行预分词之前，需要将分块中的 <|endoftext|> 移除，防止其参与后续统计
    - 等队列中所有文档分块被消费完成之后，持久化预分词结果，便于后续字节对合并迭代时流式读取
- 字节对合并迭代多进程优化
    - 使用 `_BPEStreamWorkerPool` 管理多个 worker 进程（默认为主机 CPU 核数 + 1）
    - 将 chunk 文件列表分配给各个 worker，每个 worker 负责处理自己的 chunk 文件
    - 每次 merge 操作时，各个 worker 并行处理自己的 chunk 文件，返回频率增量 delta
    - 主进程收集所有 worker 的 delta 并更新全局 pair_counts
    - 如果所有 chunk 文件都能加载到内存中（`all_in_memory=True`），则直接在内存中处理，无需多进程
- Merge 倒排索引与磁盘 offload
    - 维护字节对到 word 下标的倒排索引，每轮 merge 只处理仍含目标对的词，并在 merge 后用新旧相邻对差分更新索引，避免对全部词全量扫描
    - 流式 pretokenize 落盘时将索引与 words 一并保存；merge 时按内存阈值分批加载分块、执行合并并回写
    - 可通过 `use_inverted_index` 与 CLI 对应开关启用；训练结果与无索引路径一致，代价为略增 pretokenize 时间与 chunk 体积
- 字节对合并迭代缓存优化
    - 维护一个全局内存缓存字典 `pair_counts`，将其初始化为当前训练状态下所有字节对的全局频次；每次确定要合并的 `(left, right)` 后，不再在下一轮全量重算，而是让每个 worker 仅返回这次 merge 对字节对频次的增量 `delta`，主进程据此增量更新 `pair_counts`
    - 增量更新后要清理频次为 0 的条目，确保后续 `max(pair_counts.values())` 的语义稳定

## 编程语言
- Python：BPETrainer 和 BPETokenizer 的对外接口部分都用 Python 语言实现
- Rust：核心的执行流程以及与硬件、性能强相关的部分使用 Rust 语言实现，Rust 代码需要统一放置到 libs 目录下

## 可观测性
将 BPETrainer 的训练流程分为“数据预处理”、“数据预分词”、“字节对合并迭代”四个部分，并针对这四个部分分别进行性能指标统计：
- 数据预处理：记录“分块数量”、“预处理耗时”两个性能指标
- 数据预分词：记录“分词数量”、“预分词耗时”两个性能指标
- 字节对合并迭代： 记录“每秒任务入队列吞吐”、“每秒任务出队列吞吐”、“消费者进程平均任务处理耗时”、“每轮 merge 的平均耗时”四个性能指标

注意：BPETrainer 对应的 cli 命令需要在上述阶段开始时打印对应阶段开始的日志，并在对应阶段结束时打印统计到的性能指标；对于“字节对合并迭代”部分，每次完成一个迭代就打印对应轮次的性能指标