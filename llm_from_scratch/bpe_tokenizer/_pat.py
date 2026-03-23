import regex

# 与 tiktoken GPT-2 (r50k) 一致 — 用于 encode/decode 与 pytest+tiktoken 对齐
# 见 tiktoken_ext/openai_public.py 中 r50k_pat_str
TIKTOKEN_GPT2_PAT_STR = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)

# 课程文档 PAT — 用于 train_bpe，与仓库 train-bpe-reference-* 生成方式一致
CS336_PAT_STR = (
    r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

ENCODE_SPLIT_PATTERN = regex.compile(TIKTOKEN_GPT2_PAT_STR)
TRAIN_SPLIT_PATTERN = regex.compile(CS336_PAT_STR)
# 向后兼容旧名：编码侧
GPT2_SPLIT_PATTERN = ENCODE_SPLIT_PATTERN
