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
    - 训练数据预分词：使用 GPT2 PAT 正则规则，将原始的完整训练数据拆分成一个个语义单元，作为后续字节对的统计对象。
        - PAT 正则：`r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`
    - 合并迭代：维护一个字节对字典用于统计字节对出现的频率，遍历预处理阶段生成的语义单元，查找语义单元中的字节对并更新字节对字典中的词频信息。在完成全局统计后，
    将出现频率最高的字节对加入词汇表中。
        - 持续运行合并迭代步骤，直到词汇表达到预设大小

### BPETokenizer
- BPETokenizer 的输入是 tokenizer 文件和一个文件名，分别对应 tokenizer 词汇表以及待进行 tokenization 操作的数据文件
- BPETokenizer 

## 对外接口