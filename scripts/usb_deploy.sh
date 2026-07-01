#!/usr/bin/env bash
# scripts/usb_deploy.sh  (由 build_usb.sh 拷到 U 盘，在目标离线机上运行)
# 从 U 盘载入 whisper-stt:airgap 镜像并起服务(单 GPU 镜像，CPU/GPU 二选一)。
#
# 用法(进入 U 盘根目录):
#   bash usb_deploy.sh                       # 自动判断: 富余显存用 GPU，否则 CPU
#   bash usb_deploy.sh --cpu                 # 强制 CPU(目标机只需 Docker)
#   bash usb_deploy.sh --gpu                 # 强制 GPU(需 NVIDIA 驱动 + nvidia-ctk)
#   bash usb_deploy.sh --dir /opt/whisper-airgap   # 指定部署目录(默认 /opt/whisper-airgap)
set -euo pipefail

PROFILE=""
DEPLOY_DIR="/opt/whisper-airgap"
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) PROFILE="cpu"; shift;;
    --gpu) PROFILE="gpu"; shift;;
    --dir) DEPLOY_DIR="${2:?--dir 需要一个路径}"; shift 2;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0;;
    *) echo "未知参数: $1 (用 -h 查看帮助)"; exit 1;;
  esac
done

USB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PORT="5012"

# ---- 前置检查 ----
command -v docker >/dev/null 2>&1 || { echo "✗ 未安装 docker"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "✗ 缺少 docker compose 插件(v2)"; exit 1; }
[ -f "$USB_DIR/whisper.tar.gz" ] || { echo "✗ 当前目录找不到 whisper.tar.gz，请在 U 盘根目录运行"; exit 1; }

# ---- 自动判断 CPU / GPU ----
if [ -z "$PROFILE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && command -v nvidia-ctk >/dev/null 2>&1; then
    FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)"
    FREE_MB="${FREE_MB:-0}"
    if [ "$FREE_MB" -ge 2048 ] 2>/dev/null; then
      PROFILE="gpu"; echo "▸ 检测到 GPU，剩余显存 ${FREE_MB}MB → 选用 GPU (cuda/float16)"
    else
      PROFILE="cpu"; echo "▸ 检测到 GPU 但剩余显存仅 ${FREE_MB}MB(<2048) → 选用 CPU"
    fi
  else
    PROFILE="cpu"; echo "▸ 未检测到可用 GPU(或未装 nvidia-ctk) → 选用 CPU"
  fi
fi
echo "▸ 部署目录: $DEPLOY_DIR"
echo "▸ 推理模式: $PROFILE   (端口 ${HOST_PORT})"
echo "5 秒后开始，Ctrl-C 取消..."; sleep 5

# ---- 写部署目录(若 /opt 不可写则自动 sudo；docker 命令本身走 socket，不需要 sudo) ----
SUDO=""
if [ "$(id -u)" -ne 0 ] && [ ! -w "$(dirname "$DEPLOY_DIR")" ]; then
  SUDO="sudo"; echo "▸ 部署目录需 root 权限写入，自动使用 sudo"
fi
$SUDO mkdir -p "$DEPLOY_DIR"

echo "==> [1/4] 载入镜像 docker load (镜像较大，约 1-2 分钟)..."
docker load -i "$USB_DIR/whisper.tar.gz"

echo "==> [2/4] 拷贝 模型 + compose 到 $DEPLOY_DIR"
$SUDO rm -rf "$DEPLOY_DIR/whisper-models"
$SUDO cp -r "$USB_DIR/whisper-models" "$DEPLOY_DIR/whisper-models"
$SUDO cp    "$USB_DIR/docker-compose.airgap.yml" "$DEPLOY_DIR/docker-compose.airgap.yml"

echo "==> [3/4] 起服务 (profile=$PROFILE)..."
docker compose -f "$DEPLOY_DIR/docker-compose.airgap.yml" --profile "$PROFILE" up -d

echo "==> [4/4] 健康检查 (首次转写时会触发模型加载，稍候几秒)..."
OK=0
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:${HOST_PORT}/health" >/dev/null 2>&1; then
    OK=1; break
  fi
  sleep 2
done
if [ "$OK" = 1 ]; then
  echo "✓ 服务就绪:"; curl -s "http://localhost:${HOST_PORT}/health"; echo
else
  echo "✗ 30 秒内未就绪，看日志: docker compose -f $DEPLOY_DIR/docker-compose.airgap.yml logs"
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF

=========================================================
 部署完成。访问:  http://${IP:-localhost}:${HOST_PORT}
 健康:   curl http://localhost:${HOST_PORT}/health
 转写:   POST http://localhost:${HOST_PORT}/v1/audio/transcriptions
         (model 传 large-v3-turbo 或不传用默认)
 日志:   docker compose -f $DEPLOY_DIR/docker-compose.airgap.yml logs -f
 停止:   docker compose -f $DEPLOY_DIR/docker-compose.airgap.yml --profile $PROFILE down
=========================================================
EOF
