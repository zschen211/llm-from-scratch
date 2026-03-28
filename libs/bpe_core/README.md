# BPE Core - Rust 加速模块

这个目录包含 BPE tokenizer 的 Rust 核心实现，用于加速性能关键的操作。

## 架构设计

### Python 接口层
- `src/bpe_tokenizer/_rust_bridge.py` - Python 包装层，提供统一接口
- 自动检测 Rust 模块是否可用
- 如果 Rust 不可用，自动回退到 Python 实现
- 保持与原有 Python 接口完全兼容

### Rust 核心层
- `libs/bpe_core/src/lib.rs` - PyO3 绑定和 Python 接口
- `libs/bpe_core/src/pair_counter.rs` - pair 频率统计（Rust 实现）
- `libs/bpe_core/src/merge_optimizer.rs` - merge 操作和倒排索引（Rust 实现）
- `libs/bpe_core/src/pretokenizer.rs` - 预分词（Rust 实现）

## 性能优化

Rust 实现相比 Python 实现的优势：

1. **零成本抽象** - 编译时优化，无运行时开销
2. **内存效率** - 更紧凑的数据结构，减少内存占用
3. **并行处理** - 使用 rayon 实现数据并行
4. **类型安全** - 编译时类型检查，避免运行时错误

预期性能提升：
- `count_pairs`: 5-10x
- `merge_pair_all_words_with_deltas`: 10-20x
- `preprocess_and_pretokenize`: 3-5x

## 构建说明

### 前置要求

1. 安装 Rust（如果尚未安装）:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. 安装 maturin:
   ```bash
   pip install maturin
   ```

### 构建步骤

#### 方法 1: 使用构建脚本（推荐）

```bash
cd libs/bpe_core
./build.sh
```

#### 方法 2: 手动构建

```bash
cd libs/bpe_core
maturin develop --release
```

### 验证安装

```bash
python -c "import bpe_core; print('Rust module loaded successfully!')"
```

或者在 Python 中：

```python
from llm_from_scratch.bpe_tokenizer._rust_bridge import RUST_AVAILABLE

if RUST_AVAILABLE:
    print("Rust acceleration is enabled!")
else:
    print("Using Python fallback implementation")
```

## 开发说明

### 修改 Rust 代码后重新构建

```bash
cd libs/bpe_core
maturin develop --release
```

### 运行 Rust 测试

```bash
cd libs/bpe_core
cargo test
```

### 性能基准测试

```bash
cd libs/bpe_core
cargo bench
```

## 接口兼容性

Rust 实现完全兼容 Python 接口：

```python
# 这些函数会自动使用 Rust 实现（如果可用）
from llm_from_scratch.bpe_tokenizer import train_bpe

vocab, merges = train_bpe(
    input_path="data.txt",
    vocab_size=5000,
    special_tokens=["<|endoftext|>"],
)
```

所有现有的测试和 CLI 工具无需修改即可使用 Rust 加速。

## 故障排除

### 问题: `ImportError: No module named 'bpe_core'`

**解决方案**: 运行构建脚本
```bash
cd libs/bpe_core
./build.sh
```

### 问题: 构建失败

**可能原因**:
1. Rust 未安装或版本过旧
2. maturin 未安装
3. 编译器错误

**解决方案**:
1. 更新 Rust: `rustup update`
2. 安装 maturin: `pip install maturin`
3. 查看详细错误信息并根据提示修复

### 问题: 性能没有提升

**检查**:
1. 确认 Rust 模块已正确加载:
   ```python
   from llm_from_scratch.bpe_tokenizer._rust_bridge import RUST_AVAILABLE
   print(RUST_AVAILABLE)  # 应该输出 True
   ```

2. 确认使用了 release 构建:
   ```bash
   maturin develop --release  # 注意 --release 标志
   ```

## 技术栈

- **PyO3** - Rust 和 Python 的绑定
- **rustc-hash** - 快速哈希表实现
- **rayon** - 数据并行库
- **regex** - 正则表达式引擎
- **maturin** - Rust Python 扩展构建工具

## 贡献指南

修改 Rust 代码时：

1. 保持与 Python 接口的兼容性
2. 添加单元测试（`#[cfg(test)]` 模块）
3. 更新文档
4. 运行测试: `cargo test`
5. 检查格式: `cargo fmt`
6. 运行 linter: `cargo clippy`

## 许可证

与主项目相同。
