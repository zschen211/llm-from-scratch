#!/usr/bin/env bash
# 构建 Rust BPE 核心模块

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUST_DIR="$PROJECT_ROOT/libs/bpe_core"

echo "=== Building Rust BPE Core Module ==="
echo "Rust directory: $RUST_DIR"
echo ""

# 检查 Rust 是否安装
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# 检查 maturin 是否安装
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# 进入 Rust 目录
cd "$RUST_DIR"

# 构建 Rust 扩展
echo "Building Rust extension..."
maturin develop --release

echo ""
echo "=== Build Complete ==="
echo "The Rust module 'bpe_core' is now available for import in Python."
echo ""
echo "Test with:"
echo "  python -c 'import bpe_core; print(\"Rust module loaded successfully!\")'"
