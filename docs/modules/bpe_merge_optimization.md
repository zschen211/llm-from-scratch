# BPE Merge 优化：倒排索引 + 磁盘 Offload

## 问题分析

### 原始实现的性能瓶颈

BPE 训练的 merge 阶段存在严重的性能瓶颈：

1. **算法复杂度：O(vocab_size × total_tokens)**
   - 每次 merge 都要扫描所有 words 的所有 tokens
   - 对于 vocab_size=10000，需要执行 10000 次全量扫描
   - 即使某个 word 不包含目标 pair，也要遍历检查

2. **高频小操作累积**
   ```python
   # 每次 merge 都会执行数百万次：
   - list.append()          # 构建新 word
   - len(out)               # 获取列表长度
   - dict.get(p, 0)         # 字典查询
   - set() 转换             # 去重操作
   - tuple 创建             # 创建 pair
   ```

3. **数据结构低效**
   - 频繁的 dict.get() 而不是 defaultdict
   - 每次都创建新的 out list，没有复用
   - list + set 转换的额外开销

### 性能影响

对于 50MB 的 TinyStories 数据集：
- 原始实现：merge 阶段可能需要数小时
- 大部分时间浪费在扫描不包含目标 pair 的 words 上
- 后期 merge 时，大部分 words 已经不包含目标 pair，但仍需全量扫描

## 优化方案：倒排索引 + 磁盘 Offload

### 核心思路

**倒排索引**：维护 `pair → set[word_idx]` 的映射，只处理包含目标 pair 的 words。

**磁盘 Offload**：将索引随 chunk 一起序列化到磁盘，按需加载/卸载，避免 OOM。

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│  WordsChunkWithIndex                                        │
│  ┌────────────────┐  ┌──────────────────────────────────┐  │
│  │ words          │  │ pair_index                       │  │
│  │ list[list[    │  │ dict[tuple[bytes, bytes],        │  │
│  │   bytes]]      │  │      set[int]]                   │  │
│  └────────────────┘  └──────────────────────────────────┘  │
│                                                             │
│  Methods:                                                   │
│  - build_index()                                            │
│  - merge_pair_with_deltas(left, right, merged)             │
│  - save(path)  / load(path)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 实现细节

#### 1. 倒排索引构建

```python
def build_index(self) -> None:
    """构建倒排索引：pair → set[word_idx]。"""
    self.pair_index = defaultdict(set)
    for word_idx, word in enumerate(self.words):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            self.pair_index[pair].add(word_idx)

    # 转为普通 dict 以节省内存
    self.pair_index = dict(self.pair_index)
```

**时间复杂度**：O(total_tokens)，只需构建一次。

**空间复杂度**：约为 words 的 30-50%（取决于 pair 的重复度）。

#### 2. 使用索引加速 merge

```python
def merge_pair_with_deltas(
    self, left: bytes, right: bytes, merged: bytes
) -> dict[tuple[bytes, bytes], int]:
    """只处理包含目标 pair 的 words。"""
    target_pair = (left, right)
    affected_word_indices = self.pair_index.get(target_pair, set())

    if not affected_word_indices:
        return {}  # 快速返回，无需扫描

    # 只遍历包含目标 pair 的 words
    for word_idx in affected_word_indices:
        word = self.words[word_idx]
        # ... 执行 merge 并更新索引 ...
```

**关键优化**：
- 复杂度从 O(total_tokens) 降到 O(affected_tokens)
- 对于低频 pair，加速可达 100-1000 倍
- 后期 merge 时，大部分 words 已经不包含目标 pair，直接跳过

#### 3. 索引维护

每次 merge 后需要更新索引：

```python
# 记录旧的 pairs（merge 前）
old_pairs_in_word = {(word[i], word[i+1]) for i in range(len(word)-1)}

# 执行 merge
out = merge_word(word, left, right, merged)

# 记录新的 pairs（merge 后）
new_pairs_in_word = {(out[i], out[i+1]) for i in range(len(out)-1)}

# 更新索引
removed_pairs = old_pairs_in_word - new_pairs_in_word
added_pairs = new_pairs_in_word - old_pairs_in_word

for pair in removed_pairs:
    self.pair_index[pair].discard(word_idx)
    if not self.pair_index[pair]:
        del self.pair_index[pair]

for pair in added_pairs:
    if pair not in self.pair_index:
        self.pair_index[pair] = set()
    self.pair_index[pair].add(word_idx)
```

#### 4. 磁盘序列化

```python
def save(self, path: Path) -> None:
    """序列化到磁盘（包含 words 和索引）。"""
    payload = {
        "words": self.words,
        "pair_index": {
            # 将 set 转为 list 以便序列化
            pair: list(indices) for pair, indices in self.pair_index.items()
        },
        "index_built": self._index_built,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### 与流式模式的集成

#### Pretokenize 阶段

```python
# 保存 chunk 时同时构建索引
chunk_with_index = WordsChunkWithIndex(accumulated_words)
chunk_with_index.build_index()
_dump_words_chunk_with_index(cpath, chunk_with_index)
```

#### Merge 阶段

```python
# 加载 chunk 时同时加载索引
chunk = _load_words_chunk_with_index(cpath)

# 使用索引加速 merge
delta = chunk.merge_pair_with_deltas(left, right, merged)

# 回写 chunk（索引已更新）
_dump_words_chunk_with_index(cpath, chunk)
```

### 内存管理

#### 分批加载策略

```python
def _load_chunk_batch(
    chunk_files: list[Path],
    start_idx: int,
    memory_target_percent: float,
    use_inverted_index: bool = False,
) -> tuple[list[tuple[Path, Any]], int]:
    """按需加载 chunk，达到内存阈值时停止。"""
    loaded = []
    idx = start_idx

    while idx < len(chunk_files):
        if loaded:
            mem = _get_system_memory_percent()
            if mem >= memory_target_percent:
                break

        path = chunk_files[idx]
        if use_inverted_index:
            chunk = _load_words_chunk_with_index(path)
        else:
            chunk = _load_words_chunk(path)

        loaded.append((path, chunk))
        idx += 1

    return loaded, idx
```

#### 内存占用估算

- **words**：约 N MB（原始数据）
- **pair_index**：约 0.3-0.5 × N MB（索引开销）
- **总计**：约 1.3-1.5 × N MB

对于 50MB 的数据，索引约占 15-25 MB。

## 使用方法

### CLI 参数

```bash
# 启用倒排索引（默认）
python cli/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data.txt \
  --vocab-size 5000 \
  --special-token "<|endoftext|>" \
  --out tokenizer.json \
  --use-inverted-index

# 禁用倒排索引（用于对比）
python cli/bpe_tokenizer/train_bpe_cli.py \
  --input-corpus data.txt \
  --vocab-size 5000 \
  --special-token "<|endoftext|>" \
  --out tokenizer.json \
  --no-inverted-index
```

### Python API

```python
from llm_from_scratch.bpe_tokenizer import train_bpe

vocab, merges = train_bpe(
    input_path="data.txt",
    vocab_size=5000,
    special_tokens=["<|endoftext|>"],
    use_inverted_index=True,  # 启用倒排索引
    stream_chunk_chars=1_000_000,  # 流式模式
    stream_memory_target_percent=70,  # 内存阈值
)
```

## 性能对比

### 测试环境

- 数据集：TinyStories 50MB
- Vocab size：5000
- 硬件：8 核 CPU，16GB RAM

### 预期结果

| 指标 | 无索引 | 有索引 | 提升 |
|------|--------|--------|------|
| Merge 总时间 | ~2 小时 | ~10 分钟 | 12x |
| 内存占用 | ~500 MB | ~650 MB | +30% |
| 磁盘 I/O | 中等 | 中等 | 相当 |

### 加速原理

1. **早期 merge**：目标 pair 频率高，affected_words 多，加速不明显（1-2x）
2. **中期 merge**：目标 pair 频率中等，加速明显（5-10x）
3. **后期 merge**：目标 pair 频率低，加速显著（50-100x）

**平均加速**：约 10-20x（取决于数据特征）

## 实现文件

- `src/bpe_tokenizer/_merge_optimizer.py` - 倒排索引实现
- `src/bpe_tokenizer/train_bpe.py` - 集成到训练流程
- `cli/bpe_tokenizer/train_bpe_cli.py` - CLI 参数支持
- `evaluation/bpe-tokenizer/test_inverted_index.sh` - 性能测试脚本

## 注意事项

### 1. 内存开销

倒排索引会增加 30-50% 的内存开销。对于内存受限的环境：
- 降低 `stream_memory_target_percent`（如 60）
- 增加 `min_pair_freq`（如 2 或 3）过滤低频 pair

### 2. 索引构建时间

每个 chunk 在保存时需要构建索引，增加约 10-20% 的 pretokenize 时间。但这是一次性成本，merge 阶段的加速远超这个开销。

### 3. 磁盘空间

带索引的 chunk 文件约为原始 chunk 的 1.3-1.5 倍。确保有足够的磁盘空间。

### 4. 确定性保证

倒排索引不影响 BPE 训练的确定性。merge 顺序和结果与原始实现完全一致。

## 未来优化方向

1. **增量索引更新**：只更新受影响的 pair，而不是重建整个索引
2. **压缩索引**：使用更紧凑的数据结构（如 bitmap）
3. **并行索引构建**：多进程并行构建索引
4. **Cython 加速**：将热点函数编译为 C 扩展

## 参考资料

- [BPE 训练流程文档](./bpe_tokenizers.md)
- [倒排索引原理](https://en.wikipedia.org/wiki/Inverted_index)
- [Python pickle 性能优化](https://docs.python.org/3/library/pickle.html#performance)
