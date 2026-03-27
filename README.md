# llm-from-scratch

从零开始构建大语言模型（LLM）的教育项目，涵盖分词器（Tokenizer）、Transformer 架构、训练流程等核心模块的实现。

## QuickStart

### 环境准备

```bash
# 使用 uv 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 快速训练 BPE Tokenizer

```bash
# 使用 CLI 训练分词器
python cli/llm_from_scratch/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus tests/fixtures/corpus.en \
  --vocab-size 500 \
  --special-token "<|endoftext|>" \
  --out tokenizer.json

# 使用训练好的分词器进行编码/解码
python cli/llm_from_scratch/bpe_tokenizer/bpe_tokenizer_cli.py encode \
  --checkpoint tokenizer.json \
  --text "hello world" \
  --format json
```

### 运行测试

```bash
# 运行单元测试
uv run pytest tests/ -q

# 运行 CLI 测试
uv run pytest cli-tests/ -q

# 运行所有测试
uv run pytest tests/ cli-tests/ -q
```

## 项目介绍

本项目实现了构建 LLM 所需的核心组件：

### 核心模块

| 模块 | 功能 | 位置 |
|------|------|------|
| **BPE Tokenizer** | 字节对编码分词器，支持多进程并行训练、流式编码/解码 | `llm_from_scratch/bpe_tokenizer/` |
| **Transformer** | 包含多头注意力、前馈网络、RMSNorm、RoPE 位置编码等 | `llm_from_scratch/` |
| **训练流程** | 支持 checkpoint、梯度裁剪、学习率调度、AdamW 优化器 | `llm_from_scratch/` |
| **CLI 工具** | 命令行工具用于训练和测试各模块 | `cli/` |

### 技术特性

- **多进程并行**：BPE 训练支持预分词和合并迭代阶段的多进程并行
- **性能可观测**：内置性能指标埋点，包括队列吞吐、任务处理耗时等
- **流式处理**：`encode_iterable` 支持对大文件进行流式分词
- **测试覆盖**：pytest 单元测试 + CLI 集成测试双重保障
- **GPT-2 兼容**：分词器行为与 `tiktoken` GPT-2 编码对齐

### 项目结构

```
llm-from-scratch/
├── llm_from_scratch/          # 核心实现
│   └── bpe_tokenizer/         # BPE 分词器
│       ├── train_bpe.py       # 训练实现（支持多进程）
│       ├── codec.py           # 编解码实现
│       └── regression/        # 回归测试数据
├── cli/                       # CLI 工具
│   └── llm_from_scratch/
│       └── bpe_tokenizer/
│           ├── train_bpe_cli.py      # 训练 CLI
│           └── bpe_tokenizer_cli.py  # 编解码 CLI
├── cli-tests/                 # CLI 测试
├── tests/                     # 单元测试
├── docs/                      # 文档
│   ├── modules/               # 模块设计文档
│   └── cli/                   # CLI 使用文档
└── pyproject.toml             # 项目配置
```

## 系统架构图

### BPE Tokenizer 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                        BPE 训练流程                          │
├─────────────────────────────────────────────────────────────┤
│  1. 数据预处理                                                │
│     ├── 流式读取文档                                          │
│     ├── 按特殊 token (<|endoftext|>) 分块                    │
│     └── 输出: segments[]                                      │
│                          ↓                                    │
│  2. 数据预分词 (并行)                                         │
│     ├── 多进程队列 (num_workers)                              │
│     ├── 每进程: PAT findall → bytes                          │
│     └── 输出: words[][]                                       │
│                          ↓                                    │
│  3. 字节对合并迭代 (并行)                                     │
│     ├── 多进程队列                                            │
│     ├── 每轮: count_pairs → pick_best → merge                │
│     └── 输出: vocab, merges                                   │
└─────────────────────────────────────────────────────────────┘
```

### CLI 架构

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI 架构                             │
├─────────────────────────────────────────────────────────────┤
│  train_bpe_cli.py                                           │
│  ├── 参数解析 → train_bpe()                                  │
│  ├── 性能指标回调 (stage_start/stage_end/merge_iter_end)    │
│  └── 输出 checkpoint JSON                                     │
│                          ↓                                    │
│  bpe_tokenizer_cli.py                                         │
│  ├── encode: text → token_ids                                │
│  ├── decode: token_ids → text                                │
│  └── encode_iterable: 流式处理大文件                         │
└─────────────────────────────────────────────────────────────┘
```

## 测试说明

### 测试分层

| 测试类型 | 位置 | 说明 |
|---------|------|------|
| **单元测试** | `tests/` | 各模块函数/类的正确性测试 |
| **CLI 测试** | `cli-tests/` | 命令行工具的集成测试 |
| **回归测试** | `tests/_snapshots/` | 结果快照比对，确保行为一致性 |

### 运行特定测试

```bash
# BPE 训练测试
uv run pytest tests/test_train_bpe.py -q

# Tokenizer 编解码测试
uv run pytest tests/test_tokenizer.py -q

# 回归测试（对比参考 merges/vocab）
uv run pytest tests/test_train_bpe.py::test_train_bpe -q

# 大语料测试（5M，验证多进程正确性）
uv run pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -q
```

### Cursor 规则

项目使用 Cursor 规则进行代码规范约束：

- `public-api-cli.mdc`: 每个 public 方法需有对应 CLI
- `cli-tests.mdc`: 每个 CLI 需有对应测试
- `py-filename-lowercase-underscore.mdc`: Python 文件命名规范（仅小写+下划线）