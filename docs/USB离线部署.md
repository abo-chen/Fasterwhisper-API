# USB 离线部署（中文）

用 U 盘把自包含镜像搬到目标机离线部署，全程无需联网。**一张 GPU 镜像**覆盖 CPU/GPU 两种推理，到目标机看显存再决定用哪个。

- 镜像：`whisper-stt:airgap`（CUDA 基底，压缩后约 2GB+）
- 模型：仅 `large-v3-turbo`（约 1.6G，多语言，**法语兜底**用）
- 默认端口：`5012`
- 目标机：CPU 模式只要有 Docker；GPU 模式需 NVIDIA 驱动 + nvidia-ctk

> 配套脚本：`scripts/build_usb.sh`（开发机构建+导出）、`scripts/usb_deploy.sh`（目标机部署，随 U 盘带过去）。
> **改了代码/模型后重跑 `build_usb.sh` 即可。**

---

## 一、构建机（只做一次）

### 前提
- 项目已 clone，`whisper-models/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/` 下有 turbo 模型
- 已装 docker

### 构建 + 导出到 U 盘
```bash
cd /home/abel-chen/.local/opt/fasterwhisper
bash scripts/build_usb.sh /mnt/usb        # /mnt/usb 换成你 U 盘挂载点
```
脚本自动：构建 `whisper-stt:airgap`（不碰开发用的 `whisper-stt:latest`）→ 暂存 turbo 模型（解引用软链，FAT32 友好）→ `docker save | gzip` 到 U 盘 → 拷上 compose + 部署脚本 + 本说明。

构建机产物（U 盘上）：
```
whisper.tar.gz              # 镜像压缩包（~2GB+）
whisper-models/             # 仅 turbo 模型（~1.6G）
docker-compose.airgap.yml   # 起容器用的 compose（cpu/gpu 双 profile）
usb_deploy.sh               # 目标机一键部署脚本
部署说明.md                  # 本文档副本
```

> 只构建不导出：`bash scripts/build_usb.sh`（不带参数）。
> 镜像导出后保留在本地，不用时可 `docker rmi whisper-stt:airgap` 清理（U 盘 tar 还在，随时 `docker load` 找回）。

---

## 二、目标机（每台，约 2–4 分钟）

### 前提（按要用的模式准备）
- **共同**：Docker + docker compose 插件（v2）
- **仅 GPU 模式额外需要**：
  1. NVIDIA 驱动：`nvidia-smi` 能看到 GPU
  2. NVIDIA Container Toolkit：`nvidia-ctk --version`

### 一键部署
```bash
# 插上 U 盘，进入 U 盘目录
cd /media/$USER/你的U盘
bash usb_deploy.sh                # 自动按显存选 CPU/GPU
# 或显式指定:
bash usb_deploy.sh --cpu          # 强制 CPU
bash usb_deploy.sh --gpu          # 强制 GPU
```
脚本自动：校验 docker → `docker load` 载入镜像 → 拷贝模型+compose 到 `/opt/whisper-airgap` → `docker compose --profile up -d` → 健康检查 → 打印访问地址。

### 验证
```bash
curl http://localhost:5012/health          # 返回配置/状态 JSON
```
转写（OpenAI 兼容）：
```bash
curl http://localhost:5012/v1/audio/transcriptions \
  -F 'file=@法语音频.mp3' -F 'model=large-v3-turbo'
```
> 首次转写会触发模型加载（CPU 几秒，GPU 更快）；`model` 参数传 `large-v3-turbo` 或不传用默认。

---

## 三、CPU 还是 GPU？怎么选

部署脚本默认**自动判断**：检测到 GPU 且剩余显存 ≥ 2048MB → GPU（`cuda/float16`），否则 CPU（`cpu/int8`）。也可用 `--cpu` / `--gpu` 强制。

| | CPU 模式 | GPU 模式 |
|---|---|---|
| 目标机要求 | 仅 Docker | + NVIDIA 驱动 + nvidia-ctk |
| 速度 | ~2–5x 实时（8 核） | 更快 |
| 显存占用 | 0 | ~1.5–2GB |
| 适用 | 兜底 / 无 GPU 机器 | 有富余显存时 |

> 这是兜底模块：英/中/粤已由别的服务处理，本模块主要跑**法语**。显存被主服务占满时，自动落到 CPU，不影响主服务。

---

## 四、常见配置

### 改对外端口
编辑 `/opt/whisper-airgap/docker-compose.airgap.yml`，把 `ports: - "5012:8000"` 的**第一个** `5012` 改成想要的端口。

### 换模型
当前只打包了 `large-v3-turbo`。如需换模型：在开发机把对应模型放进 `whisper-models/hub/`，改 compose 的 `WHISPER_MODEL`，重跑 `build_usb.sh`。

### 开机自启
已设 `restart: unless-stopped`，机器重启自动拉起。

### 部署到别的目录
`bash usb_deploy.sh --dir /srv/whisper`（默认 `/opt/whisper-airgap`；写 `/opt` 需 sudo，脚本自动加）。

---

## 五、排错

| 现象 | 排查 |
|---|---|
| `--gpu` 起不来 / 报 GPU 不可用 | 目标机没装 `nvidia-ctk`，或驱动版本不够；先用 `--cpu` 兜底 |
| 健康检查 30 秒未就绪 | `docker compose -f /opt/whisper-airgap/docker-compose.airgap.yml logs` 看日志 |
| 报模型找不到 / 加载失败 | 确认 `/opt/whisper-airgap/whisper-models/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/` 存在且非空；`HF_HUB_OFFLINE=1` 已强制本地 |
| `docker load` 慢 | 解压 2GB+ 镜像是 CPU 活，1–2 分钟正常，不是 U 盘瓶颈 |
| 端口被占 | 改 compose 的 `ports` 第一个 5012 |
| 客户端传 `model=whisper-1` 报错 | 离线只内置了 `large-v3-turbo`；让客户端传 `large-v3-turbo` 或不传 |

---

## 六、更新镜像（代码/模型改了）

构建机：
```bash
bash scripts/build_usb.sh /mnt/usb      # 重做 U 盘
```
目标机：
```bash
bash usb_deploy.sh --cpu   # 或 --gpu；docker load 覆盖同名镜像，up -d 用新镜像重建容器
```

---

## 附
- 镜像与开发机 GPU 镜像同源（同一 Dockerfile + CUDA 基底），构建过程不影响 `whisper-stt:latest` 及其运行中的容器。
- 性能参考：`large-v3-turbo` 在 CPU(int8) 约 2–5x 实时，GPU(float16) 更快；显存峰值约 1.5–2GB。
