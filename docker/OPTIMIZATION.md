# Docker 镜像构建优化说明

## 优化前的主要问题

1. **Rust 编译慢**：每次都从头编译 bpe_core，没有缓存 Cargo 依赖
2. **Python 依赖安装慢**：使用 pip 而非 uv（项目本地使用 uv）
3. **apt 包重复下载**：没有使用 BuildKit cache mount
4. **层缓存不够精细**：一次性复制所有文件，导致小改动触发大范围重建

## 优化措施

### 1. Rust 编译优化（最大提速）

**优化前：**
```dockerfile
COPY libs/bpe_core /workspace/libs/bpe_core
RUN pip install -U pip maturin && \
    cd /workspace/libs/bpe_core && \
    maturin build --release
```

**优化后：**
```dockerfile
# 先复制 Cargo 依赖文件
COPY --link libs/bpe_core/Cargo.toml libs/bpe_core/Cargo.lock /workspace/libs/bpe_core/

# 预安装 maturin
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir maturin

# 再复制源码
COPY --link libs/bpe_core/src /workspace/libs/bpe_core/src

# 使用 Cargo cache mount 缓存依赖
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/workspace/libs/bpe_core/target \
    cd /workspace/libs/bpe_core && maturin build --release
```

**效果：**
- Cargo 依赖（如 pyo3、regex 等）只下载一次，后续构建复用缓存
- 源码改动不会触发依赖重新下载
- 预计 Rust 编译时间从 2-5 分钟降至 10-30 秒（首次仍需完整编译）

### 2. Python 依赖安装优化

**优化前：**
```dockerfile
RUN pip install -U pip setuptools wheel && \
    python -c "import tomllib; ..." && \
    pip install -r /tmp/requirements-thirdparty.txt
```

**优化后：**
```dockerfile
# 安装 uv
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

# 使用 uv 安装依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --no-hashes --no-dev > /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt
```

**效果：**
- uv 比 pip 快 10-100 倍（并行下载、更快的依赖解析）
- 直接读取 uv.lock，确保版本一致性
- 预计依赖安装时间从 1-2 分钟降至 5-15 秒

### 3. apt 包缓存优化

**优化前：**
```dockerfile
RUN apt-get update && apt-get install -y ... && rm -rf /var/lib/apt/lists/*
```

**优化后：**
```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y ...
```

**效果：**
- apt 包缓存在多次构建间复用
- 预计 apt 安装时间从 30-60 秒降至 5-10 秒

### 4. COPY 命令优化

**优化前：**
```dockerfile
COPY pyproject.toml uv.lock /workspace/
COPY src /workspace/src
```

**优化后：**
```dockerfile
COPY --link pyproject.toml uv.lock /workspace/
COPY --link src /workspace/src
```

**效果：**
- `--link` 标志使用更高效的文件系统链接
- 减少镜像层的数据复制开销

## 构建命令

确保使用 BuildKit（Docker 18.09+）：

```bash
# 方式 1：环境变量
export DOCKER_BUILDKIT=1
docker build -t llm-sandbox -f docker/Dockerfile.sandbox .

# 方式 2：直接使用 docker buildx
docker buildx build -t llm-sandbox -f docker/Dockerfile.sandbox .
```

## 预期效果

| 阶段 | 优化前 | 优化后（首次） | 优化后（增量） |
|------|--------|----------------|----------------|
| Rust 编译 | 2-5 分钟 | 2-5 分钟 | 10-30 秒 |
| Python 依赖 | 1-2 分钟 | 5-15 秒 | 5-15 秒 |
| apt 安装 | 30-60 秒 | 5-10 秒 | 5-10 秒 |
| **总计** | **4-8 分钟** | **3-6 分钟** | **20-60 秒** |

**注意：**
- 首次构建仍需完整编译，但后续增量构建会快得多
- 修改 Python 源码（src/）不会触发依赖重装
- 修改 Rust 源码（libs/bpe_core/src/）不会重新下载 Cargo 依赖

## 进一步优化建议

1. **预构建 Rust 扩展**：如果 bpe_core 不常变动，可以预先构建 wheel 并上传到私有 PyPI
2. **多阶段并行构建**：使用 `docker buildx` 的并行构建特性
3. **镜像缓存**：使用 `--cache-from` 从远程镜像仓库拉取缓存层
