# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

从零开始构建大语言模型（LLM）的教育项目，核心模块包括 BPE Tokenizer、Transformer 架构、训练流程等。项目使用 Python 3.12+，依赖管理使用 `uv`。

## 开发环境设置

```bash
# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

## 测试命令

```bash
# 运行所有测试
uv run pytest tests/ cli-tests/ -q

# 运行单元测试
uv run pytest tests/ -q

# 运行 CLI 测试
uv run pytest cli-tests/ -q

# 运行特定测试文件
uv run pytest tests/test_train_bpe.py -q

# 运行特定测试函数
uv run pytest tests/test_train_bpe.py::test_train_bpe -q
```

## 项目架构

### 目录结构

- `src/` - 核心实现代码（包名为 `llm_from_scratch`）
  - `bpe_tokenizer/` - BPE 分词器实现
  - `sandbox/` - Docker 沙箱运行器
- `cli/` - 命令行工具（每个 public API 都有对应 CLI）
- `cli-tests/` - CLI 集成测试
- `tests/` - 单元测试
  - `fixtures/` - 测试数据
  - `_snapshots/` - 回归测试快照
- `docs/` - 设计文档

### 包结构说明

项目使用 setuptools，`src/` 目录映射到 `llm_from_scratch` 包：
- 导入路径：`from llm_from_scratch.bpe_tokenizer import train_bpe`
- 源文件路径：`src/bpe_tokenizer/train_bpe.py`

## BPE Tokenizer 架构

### 核心模块

1. **train_bpe.py** - BPE 训练实现
   - `train_bpe()` - 主训练函数，支持多进程并行、流式处理
   - 三阶段流程：数据预处理 → 数据预分词 → 字节对合并迭代
   - 统一使用流式模式：分块读取、内存占用达阈值时落盘

2. **codec.py** - 编解码实现
   - `BPETokenizer` 类：支持 `encode()`, `decode()`, `encode_iterable()`
   - 与 GPT-2 / tiktoken 行为对齐

3. **_pat.py** - 预分词正则表达式（与 tiktoken 对齐）

4. **_gpt2_bytes.py** - GPT-2 字节编码映射

### 多进程并行架构

- **预分词阶段**：文档分块通过队列分发给多个 worker 进程并行处理
- **合并迭代阶段**：使用 `_BPEStreamWorkerPool` 管理持久化 worker
  - 将 chunk 文件列表分配给各个 worker
  - 每个 worker 负责处理自己的 chunk 文件
  - 通过轻量命令控制：`"count"` / `("merge_delta", l, r, m)` / `"stop"`
  - 增量更新优化：每次 merge 只返回频率增量 delta，避免全量重算

### 流式处理机制

流式模式统一用于所有训练：
1. 顺序读取文件，按 `stream_chunk_chars` 字符分块
2. 在 special token 边界对齐，避免截断
3. 预分词后累积到内存，达到 `stream_memory_target_percent` 阈值时落盘
4. 合并迭代时分批加载 chunk 文件，处理后回写

## CLI 工具使用

### BPE Tokenizer 训练

```bash
python cli/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus tests/fixtures/corpus.en \
  --vocab-size 500 \
  --special-token "<|endoftext|>" \
  --out tokenizer.json
```

### BPE Tokenizer 编解码

```bash
# 编码
python cli/bpe_tokenizer/bpe_tokenizer_cli.py encode \
  --checkpoint tokenizer.json \
  --text "hello world" \
  --format json

# 解码
python cli/bpe_tokenizer/bpe_tokenizer_cli.py decode \
  --checkpoint tokenizer.json \
  --token-ids "[1,2,3]" \
  --format text
```

## 代码规范（Cursor Rules）

### 1. Public API 与 CLI 同步
- 每个 public 方法必须有对应 CLI（位于 `cli/` 目录）
- CLI 文件命名：`<ClassName>_cli.py` 或 `<module>_cli.py`
- 目录结构与 `src/` 包结构一致

### 2. CLI 测试覆盖
- 每个 CLI 必须在 `cli-tests/` 中有对应测试
- 测试命名：`test_<cli_name>.py`
- 必须覆盖成功路径和失败场景（缺参、非法参数、边界条件）
- 修改 CLI 或 public API 后必须运行并更新测试

### 3. Python 文件命名
- `cli/` 和 `cli-tests/` 下的 `.py` 文件名只能包含小写字母和下划线
- 禁止大写字母、驼峰命名、连字符

## 回归测试

项目使用快照测试确保行为一致性：
- BPE 训练结果快照：`tests/_snapshots/test_train_bpe_special_tokens.pkl`
- Transformer 模块快照：`tests/_snapshots/*.npz`
- 包内回归数据：`src/bpe_tokenizer/regression/`（corpus.en 的参考结果）

修改核心算法后，需要验证快照是否需要更新。

## 性能分析

BPE 训练支持 cProfile 性能分析：

```bash
python cli/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data.txt \
  --vocab-size 5000 \
  --special-token "<|endoftext|>" \
  --out tokenizer.json \
  --profile \
  --profile-dir .prof/
```

生成的 `.prof` 文件可用 `snakeviz` 或 `py-spy` 分析。

## 关键实现细节

### BPE 训练的确定性
- 预分词使用固定正则表达式（与 tiktoken 对齐）
- 字节对选择：频率最高，并列时取字典序最大
- 多进程模式下保证结果与串行模式一致

### Special Token 处理
- Special token（如 `<|endoftext|>`）不参与字节对统计和合并
- 在预处理阶段按 special token 切分文本
- 编码时优先匹配 special token

### Checkpoint 机制
- 训练过程中每次 merge 后可写入 checkpoint（JSON 格式）
- 流式模式暂不支持 checkpoint 恢复
- Checkpoint 包含 vocab 和 merges 完整状态

## 常见问题

### 内存不足
- 调整分块大小：`--stream-chunk-chars 500000`（默认 1000000）
- 调整内存阈值：`--stream-memory-target-percent 70`（默认 85）
- 减少并行度：`--num-workers 2`

### 性能优化
- 流式模式统一用于所有训练
- 如果所有 chunk 能加载到内存，自动在内存中处理
- 可通过 `--num-workers 1` 强制串行（便于调试）

### 测试失败
- 回归测试失败：检查算法是否变更，确认是否需要更新快照
- CLI 测试失败：确保 CLI 参数与 public API 签名同步
