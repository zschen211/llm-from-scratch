# BPE Tokenizer CLI

本节介绍三个 CLI 脚本：

1. `train_bpe_cli.py`：训练 vocab/merges 并写出 checkpoint JSON
2. `bpe_tokenizer_cli.py`：基于 checkpoint（或 GPT-2 vocab/merges）执行 `encode` / `decode` / `encode_iterable`
3. `train_bpe_flamegraph_cli.py`：将训练产生的 `.prof` 文件转换为 SVG 火焰图

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

| 参数 | 说明 |
|------|------|
| `--special-token` | 可重复传入；特殊 token 会被当作 encode 时的原子字符串匹配 |
| `--num-workers` | 并行 worker 数（想禁用并行可设为 `1`） |
| `--disable-packaged-regression` | 调试用，禁用 `corpus.en` 的 packaged regression 快速路径 |
| `--no-print-metrics` | 禁用逐轮性能指标输出（静默模式） |
| `--checkpoint-path` | 训练过程中写入的增量 checkpoint 路径（断点续训） |
| `--force-restart` | 忽略已有 checkpoint，强制重新训练 |
| `--profile` | 启用 cProfile 性能采样，`.prof` 文件默认写入 `.prof/` 目录 |
| `--profile-dir DIR` | 自定义 `.prof` 输出目录（隐式启用 `--profile`） |

### 性能分析（Profiling）

训练时加 `--profile` 即可对全流程进行 cProfile 采样：

```bash
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data/corpus.en \
  --vocab-size 500 \
  --special-token "<|endoftext|>" \
  --out tok.json \
  --profile
```

完成后会输出类似 `[profile] saved: .prof/train_bpe_20260323_142000_12345.prof`。

也可指定输出目录：

```bash
  --profile --profile-dir /tmp/my_profiles
```

`.prof/` 目录已加入 `.gitignore`，不会被提交。

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

## 3) 火焰图可视化：`train_bpe_flamegraph_cli.py`

将 `--profile` 产生的 `.prof` 文件转换为交互式 HTML 火焰图（基于 `pstats` + `d3-flame-graph`，无额外依赖）。

基本用法：

```bash
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py \
  .prof/train_bpe_20260323_142000_12345.prof
```

默认输出与 `.prof` 同名的 `.html` 文件（如 `.prof/train_bpe_20260323_142000_12345.html`）。

指定输出路径并在浏览器中打开：

```bash
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py \
  .prof/train_bpe_20260323_142000_12345.prof \
  --out flamegraph.html \
  --open
```

| 参数 | 说明 |
|------|------|
| `prof_file`（位置参数） | `.prof` 文件路径 |
| `--out` / `-o` | 输出 HTML 路径，默认与输入同名 `.html` |
| `--open` | 生成后用系统默认浏览器打开 |

### 完整工作流示例

```bash
# 1. 训练 + 采样
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data/corpus.en \
  --vocab-size 500 \
  --special-token "<|endoftext|>" \
  --out tok.json \
  --profile

# 2. 生成火焰图并查看
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_flamegraph_cli.py \
  .prof/train_bpe_*.prof \
  --open
```

