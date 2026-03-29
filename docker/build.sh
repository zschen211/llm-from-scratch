#!/usr/bin/env bash
#
# Docker 镜像构建脚本
#
# 构建流程：
# 0. （若缺失）在宿主机导出 requirements 并下载三方 wheel，供镜像内离线 pip 安装
# 1. 本机编译 Rust 扩展 bpe_core（使用 maturin），生成 wheel
# 2. 将 wheel 注入 Docker 构建上下文，构建镜像（容器内不需要 Rust 工具链）
#
# docker build 使用主机网络（见下方 DOCKER_BUILD_NETWORK），与宿主机 DNS/代理一致。
#
# 用法：
#   ./docker/build.sh [--no-cache] [--tag TAG]
#
# 选项：
#   --no-cache         不使用 Docker 缓存（强制重新构建所有层）
#   --tag TAG          指定镜像标签（默认：llm-from-scratch-sandbox:latest）
#

set -euo pipefail

# docker build 网络模式：使用主机网络栈（勿改 Dockerfile，由 CLI 指定）
DOCKER_BUILD_NETWORK=(--network host)

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 解析参数
DOCKER_BUILD_ARGS=()
IMAGE_TAG="llm-from-scratch-sandbox:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache)
      DOCKER_BUILD_ARGS+=(--no-cache)
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
      echo "用法: $0 [--no-cache] [--tag TAG]"
      exit 1
      ;;
  esac
done

# uv export 会带「-e .」；Docker 第一层尚无 src，且离线装 -r 会触发 PEP517 却缺 wheel 包
strip_editable_root_requirement() {
  local REQ_FILE="docker/requirements-thirdparty.txt"
  [ -f "$REQ_FILE" ] || return 0
  if ! grep -qE '^-e[[:space:]]+\.' "$REQ_FILE"; then
    return 0
  fi
  echo "从 $REQ_FILE 移除「-e .」（镜像内由后续层 pip install -e . 安装项目）"
  grep -vE '^-e[[:space:]]+\.' "$REQ_FILE" > "${REQ_FILE}.tmp" && mv "${REQ_FILE}.tmp" "$REQ_FILE"
}

# 旧缓存可能只有运行时依赖 whl，缺 PyPI 包「wheel」的 .whl；镜像最后一层离线 pip install -e . 会失败
ensure_pep517_wheels_in_thirdparty() {
  local OUT_DIR="$PWD/docker/wheels/thirdparty"
  mkdir -p "$OUT_DIR"
  shopt -s nullglob
  local _w=( "$OUT_DIR"/wheel-*.whl )
  shopt -u nullglob
  [ "${#_w[@]}" -gt 0 ] && return 0

  if ! command -v uv &> /dev/null; then
    echo "错误: $OUT_DIR 缺少 wheel-*.whl，且未找到 uv，无法补充下载。请删除 thirdparty 后重试或安装 uv。"
    exit 1
  fi
  echo "未找到 $OUT_DIR/wheel-*.whl，正在联网补充下载 setuptools、wheel（供 Docker 离线 pip install -e .）..."
  local PY
  PY="$(uv python find 3.12)"
  if ! "$PY" -m pip --version &>/dev/null; then
    uv pip install --python "$PY" pip
  fi
  "$PY" -m pip download "setuptools>=61" "wheel" -d "$OUT_DIR"
}

download_thirdparty_wheels() {
  if ! command -v uv &> /dev/null; then
    echo "错误: 未找到 uv，请先安装：https://github.com/astral-sh/uv"
    exit 1
  fi
  local OUT_DIR="$PWD/docker/wheels/thirdparty"
  local REQ_FILE="$PWD/docker/requirements-thirdparty.txt"
  mkdir -p "$OUT_DIR"

  echo "=== 导出依赖列表（uv.lock，不含 dev）==="
  uv export --frozen --no-dev --no-hashes -o "$REQ_FILE"
  strip_editable_root_requirement
  echo "已写入: $REQ_FILE（供 Docker 仅传递依赖，无 -e .）"

  echo ""
  echo "=== 下载 wheel 到 $OUT_DIR ==="
  echo "（若含 torch，体积与下载时间可能较大）"
  # uv 0.8+ 已移除 `uv pip download`，改用标准 pip download
  local PY
  PY="$(uv python find 3.12)"
  if ! "$PY" -m pip --version &>/dev/null; then
    echo "当前 Python 未提供 pip，正在安装 pip（仅用于下载 wheel）..."
    uv pip install --python "$PY" pip
  fi
  "$PY" -m pip download \
    -r "$REQ_FILE" \
    -d "$OUT_DIR"
  # PEP 517 构建依赖不在 uv export 的运行时树里；离线 pip install -e . 需要 wheel/setuptools 的 .whl
  echo "=== 补充下载 pyproject [build-system] 所需 wheel（setuptools、wheel）==="
  "$PY" -m pip download "setuptools>=61" "wheel" -d "$OUT_DIR"

  echo ""
  echo "=== 三方 wheel 下载完成 ==="
}

echo "=========================================="
echo "阶段 0: 下载三方依赖 wheel（宿主机联网，容器内离线安装）"
echo "=========================================="

shopt -s nullglob
_thirdparty_wheels=(docker/wheels/thirdparty/*.whl)
shopt -u nullglob

if [ ! -f docker/requirements-thirdparty.txt ] || [ "${#_thirdparty_wheels[@]}" -eq 0 ]; then
  echo "未找到 docker/requirements-thirdparty.txt 或 docker/wheels/thirdparty/*.whl，开始下载..."
  download_thirdparty_wheels
else
  echo "已存在第三方 wheel，跳过下载（如需刷新请删除 docker/wheels/thirdparty/*.whl 与 docker/requirements-thirdparty.txt 后重试）"
fi
# 跳过下载时仍可能沿用旧版 export（含 -e .），每次构建前统一剥掉
strip_editable_root_requirement
ensure_pep517_wheels_in_thirdparty

echo ""
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

# 编译 Rust 扩展（与 Dockerfile 中 python:3.12-slim 一致，避免误用本机默认 3.13）
cd libs/bpe_core
PY312="$(cd ../.. && uv python find 3.12)"
if [ -z "$PY312" ] || [ ! -x "$PY312" ]; then
  echo "错误: 需要可执行的 Python 3.12（uv python find 3.12），与沙箱镜像版本一致"
  exit 1
fi
echo "正在编译 bpe_core (release 模式)，解释器: $PY312"
maturin build --release --interpreter "$PY312"

shopt -s nullglob
_candidates=("$(pwd)/target/wheels/bpe_core-"*cp312*.whl)
shopt -u nullglob
if [ "${#_candidates[@]}" -eq 0 ]; then
  echo "错误: 未找到 bpe_core-*cp312*.whl，请检查 maturin 输出"
  exit 1
fi
# 优先 manylinux（Linux 容器可装）；勿用 head -n1 以免选中 target/wheels 里残留的旧平台 wheel
WHEEL=""
while IFS= read -r f; do
  [[ "$f" == *manylinux* ]] && WHEEL="$f" && break
done < <(ls -t "${_candidates[@]}" 2>/dev/null)
if [ -z "$WHEEL" ]; then
  WHEEL=$(ls -t "${_candidates[@]}" 2>/dev/null | head -n1)
  echo "警告: 未找到 manylinux 的 cp312 wheel，已选用: $WHEEL"
  echo "  若在 macOS 上构建，Linux 容器内 pip 可能拒绝安装；请在 Linux/WSL 上执行 build.sh 或配置交叉编译。"
fi

echo "✓ 编译完成: $WHEEL"

# 保留 PEP 427 文件名（勿改名为 bpe_core.whl，否则 pip 报 invalid wheel filename）
cd ../..
mkdir -p docker/wheels
rm -f docker/wheels/bpe_core-*.whl
cp "$WHEEL" "docker/wheels/$(basename "$WHEEL")"
echo "✓ wheel 已复制到 docker/wheels/$(basename "$WHEEL")"

echo ""
echo "=========================================="
echo "阶段 2: 构建 Docker 镜像"
echo "=========================================="

# 构建 Docker 镜像
echo "正在构建镜像: $IMAGE_TAG"
echo "docker build 网络: ${DOCKER_BUILD_NETWORK[*]}"
echo "构建参数: ${DOCKER_BUILD_ARGS[*]}"

DOCKER_BUILDKIT=1 docker build \
  "${DOCKER_BUILD_NETWORK[@]}" \
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
echo "运行示例（与构建一致，使用主机网络）:"
echo "  docker run --rm -it --network host $IMAGE_TAG"
