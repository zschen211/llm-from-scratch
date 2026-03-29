#!/usr/bin/env bash
#
# Docker 镜像构建脚本
#
# 两阶段构建：
# 1. 本机编译 Rust 扩展 bpe_core（使用 maturin）
# 2. 构建 Docker 镜像，注入编译好的 wheel
#
# 用法：
#   ./docker/build.sh [--no-cache] [--install-perf=0|1] [--tag TAG]
#
# 选项：
#   --no-cache         不使用 Docker 缓存（强制重新构建所有层）
#   --install-perf=0   跳过 linux-perf 安装（加快构建）
#   --install-perf=1   安装 linux-perf（默认）
#   --tag TAG          指定镜像标签（默认：llm-from-scratch-sandbox:latest）
#

set -euo pipefail

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 解析参数
DOCKER_BUILD_ARGS=()
INSTALL_PERF=1
IMAGE_TAG="llm-from-scratch-sandbox:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache)
      DOCKER_BUILD_ARGS+=(--no-cache)
      shift
      ;;
    --install-perf=0)
      INSTALL_PERF=0
      shift
      ;;
    --install-perf=1)
      INSTALL_PERF=1
      shift
      ;;
    --tag)
      shift
      IMAGE_TAG="$1"
      shift
      ;;
    --tag=*)
      IMAGE_TAG="${1#*=}"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--no-cache] [--install-perf=0|1] [--tag TAG]"
      exit 1
      ;;
  esac
done

DOCKER_BUILD_ARGS+=(--build-arg "INSTALL_PERF=$INSTALL_PERF")

echo "=========================================="
echo "阶段 1: 本机编译 Rust 扩展 bpe_core"
echo "=========================================="

# 检查 maturin 是否安装
if ! command -v maturin &> /dev/null; then
  echo "错误: 未找到 maturin，请先安装："
  echo "  uv tool install maturin"
  echo "  或"
  echo "  pip install maturin"
  exit 1
fi

# 编译 Rust 扩展
cd libs/bpe_core
echo "正在编译 bpe_core (release 模式)..."
maturin build --release

# 查找生成的 wheel（使用绝对路径）
WHEEL=$(ls "$(pwd)/target/wheels/bpe_core-"*.whl 2>/dev/null | head -n1)
if [ -z "$WHEEL" ]; then
  echo "错误: 未找到编译生成的 wheel 文件"
  exit 1
fi

echo "✓ 编译完成: $WHEEL"

# 复制 wheel 到 docker/wheels/ 目录
cd ../..
mkdir -p docker/wheels
cp "$WHEEL" docker/wheels/bpe_core.whl
echo "✓ wheel 已复制到 docker/wheels/bpe_core.whl"

echo ""
echo "=========================================="
echo "阶段 2: 构建 Docker 镜像"
echo "=========================================="

# 构建 Docker 镜像
echo "正在构建镜像: $IMAGE_TAG"
echo "构建参数: ${DOCKER_BUILD_ARGS[*]}"

DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.sandbox \
  -t "$IMAGE_TAG" \
  "${DOCKER_BUILD_ARGS[@]}" \
  .

echo ""
echo "=========================================="
echo "✓ 构建完成"
echo "=========================================="
echo "镜像标签: $IMAGE_TAG"
echo ""
echo "运行示例:"
echo "  docker run --rm -it $IMAGE_TAG"
