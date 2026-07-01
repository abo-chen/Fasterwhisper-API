#!/usr/bin/env bash
# scripts/build_usb.sh
# 在开发机构建 whisper-stt:airgap(GPU 镜像，CUDA 基底)并打包到 U 盘，用于离线部署。
# 只打包 large-v3-turbo 模型(约 1.6G)。镜像 tag 为 whisper-stt:airgap，不碰 whisper-stt:latest。
#
# 用法:
#   bash scripts/build_usb.sh               # 只构建镜像(不导出)
#   bash scripts/build_usb.sh /mnt/usb      # 构建 + 导出到 U 盘挂载点
set -euo pipefail

IMAGE="whisper-stt:airgap"
BASE_IMAGE="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
TURBO_SRC="whisper-models/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USB_DIR="${1:-}"

cd "$PROJECT_DIR"

# 前置检查：turbo 模型必须在
if [ ! -d "$TURBO_SRC/snapshots" ]; then
  echo "✗ 找不到 turbo 模型: $TURBO_SRC/snapshots"
  echo "  请先在本机准备好 whisper-models/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
  exit 1
fi

echo "==> [1/4] 构建镜像 $IMAGE (BASE_IMAGE=$BASE_IMAGE)"
echo "    (与开发机 GPU 镜像同一套 Dockerfile，不影响 whisper-stt:latest)"
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$IMAGE" \
  -f Dockerfile .

echo "==> [2/4] 暂存 turbo 模型(保留 HF 缓存结构，解引用软链为真实文件)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
STAGE_REPO="$STAGE/whisper-models/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
mkdir -p "$STAGE_REPO"
# snapshots/ 下是软链指向 blobs/；-L 解引用成真实文件，U 盘(FAT32/exFAT)也能直接拷
cp -rL "$TURBO_SRC/snapshots" "$STAGE_REPO/snapshots"
cp -rL "$TURBO_SRC/refs"      "$STAGE_REPO/refs"
echo "    模型大小: $(du -sh "$STAGE/whisper-models" | cut -f1)"

# 未指定 U 盘路径：只构建，不导出
if [ -z "$USB_DIR" ]; then
  echo
  echo "==> 镜像已构建: $IMAGE"
  echo "==> 未指定 U 盘路径，跳过导出。"
  echo "    导出请运行: bash scripts/build_usb.sh /mnt/usb"
  echo "    (临时暂存已自动清理；镜像保留在本地，不用时可 docker rmi $IMAGE)"
  exit 0
fi

echo "==> [3/4] 导出到 U 盘: $USB_DIR"
mkdir -p "$USB_DIR"
echo "    - docker save | gzip (镜像约 2GB+，请耐心等待)..."
docker save "$IMAGE" | gzip > "$USB_DIR/whisper.tar.gz"

echo "    - 拷贝 模型 / compose / 部署脚本 / 说明"
rm -rf "$USB_DIR/whisper-models"
cp -r  "$STAGE/whisper-models" "$USB_DIR/whisper-models"
cp     "$PROJECT_DIR/docker-compose.airgap.yml" "$USB_DIR/docker-compose.airgap.yml"
cp     "$SCRIPT_DIR/usb_deploy.sh"              "$USB_DIR/usb_deploy.sh"
cp     "$PROJECT_DIR/docs/USB离线部署.md"        "$USB_DIR/部署说明.md"

echo "==> [4/4] 完成。U 盘内容:"
ls -lh "$USB_DIR"
cat <<EOF

 镜像(本地保留): $IMAGE   (不用时可 docker rmi $IMAGE 清理)
 下一步: 把 U 盘插到目标机，进入 U 盘目录运行:
         bash usb_deploy.sh            # 自动按显存选 CPU/GPU
         bash usb_deploy.sh --cpu      # 或强制 CPU
EOF
