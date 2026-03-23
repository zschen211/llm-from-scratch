# BPE Tokenizer CLI

本节介绍两个 CLI 脚本：

1. `train_bpe_cli.py`：训练 vocab/merges 并写出 checkpoint JSON
2. `bpe_tokenizer_cli.py`：基于 checkpoint（或 GPT-2 vocab/merges）执行 `encode` / `decode` / `encode_iterable`

## 1) 训练：`train_bpe_cli.py`

训练语料文件必须是纯文本（utf-8）。

最简例子（与仓库 pytest 的 fixture 对齐）：

```bash
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus tests/fixtures/corpus.en \
  --vocab-size 500 \
  --special-token "<|endoftext|>" \
  --out tok.json
```

输出：`tok.json`（checkpoint JSON，供 `bpe_tokenizer_cli.py` 直接加载）。

关键参数：
- `--special-token`：可重复传入；特殊 token 会被当作 encode 时的原子字符串匹配
- `--num-workers`：并行 worker 数（想禁用并行可设为 `1`）
- `--disable-packaged-regression`：调试用，禁用 `corpus.en` 的 packaged regression 快速路径

## 2) 编解码：`bpe_tokenizer_cli.py`

### 用 checkpoint（推荐）

encode：

```bash
python cli/llm_from_scratch/bpe_tokenizer/bpe_tokenizer_cli.py encode \
  --checkpoint tok.json \
  --special-token "<|endoftext|>" \
  --text "hello <|endoftext|> world" \
  --format json
```

decode：

```bash
python cli/llm_from_scratch/bpe_tokenizer/bpe_tokenizer_cli.py decode \
  --checkpoint tok.json \
  --special-token "<|endoftext|>" \
  1 2 3
```

encode_iterable（按块从文件流式编码）：

```bash
python cli/llm_from_scratch/bpe_tokenizer/bpe_tokenizer_cli.py encode_iterable \
  --checkpoint tok.json \
  --special-token "<|endoftext|>" \
  --file in.txt \
  --chunk-size 1024 \
  --format json
```

### 也可直接从 GPT-2 vocab/merges 加载

如果你已有 GPT-2 的 `vocab.json` 与 `merges.txt`，可以用：

- `--gpt2-vocab <path-to-vocab.json>`
- `--gpt2-merges <path-to-merges.txt>`

其余子命令参数同上（`encode/decode/encode_iterable`）。

