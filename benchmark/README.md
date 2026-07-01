# Faster-Whisper large-v3-turbo 性能基准

对比 fp16 GPU / int8 GPU / CPU 三种模式在中文/英文/粤语 × 10/25/45s 上的耗时与内存。

## 目录结构

```
benchmark/
├── audio/                # TTS 素材 (Fun-CosyVoice :9880 生成 + ffmpeg 切分)
│   ├── manifest.json     # 9 条音频的元数据 (语言/时长/采样率/voice)
│   ├── *_full.wav        # 每语言一条 ~55s 源语音
│   └── {zh,en,yue}_{10,25,45}s.wav   # 精确切分后的测试音频 (24kHz mono)
├── results/              # 各模式原始结果 JSON
│   ├── fp16_gpu.json     # ✓ 完整
│   ├── int8_gpu.json     # ⚠ 不支持 (Blackwell int8 cuBLAS)
│   └── cpu.json          # ✓ 完整
├── logs/                 # 逐次推理日志 + 构建/运行日志
├── gen_audio.py          # TTS 生成 + ffmpeg 切分
├── benchmark.py          # 单模式基准 (预热 + 3 次取中位)
├── run_benchmark.sh      # 3 模式一键运行
├── Dockerfile.bench      # 基于 whisper-stt 镜像 + pynvml/soundfile
└── report.html           # 最终 HTML 报告
```

## 关键结论

### 引擎层（直连 faster-whisper 库，beam_size=5）
- **fp16 GPU**：RTX 5060 Ti 上 RTF ≈ 0.05–0.13（实时 8–20×），45s 中文 ~3.0s。峰值显存 ~2.5GB。
- **int8 GPU**：**本机不支持**。RTX 5060 Ti (Blackwell sm_120) 上 ctranslate2 4.7.1 的 cuBLAS
  不支持 int8 GEMM，`int8 / int8_float16 / int8_float32` 均在首次推理抛
  `CUBLAS_STATUS_NOT_SUPPORTED`。需升级 ctranslate2 / 换 Blackwell int8 支持的构建。
- **CPU int8**：RTF ≈ 0.12–0.40，45s 中文 ~10.8s；峰值 RSS ~2.1GB。短音频受固定开销拖累明显。

### API 层（HTTP 调用运行中的服务，beam_size=1）
- **GPU fp16**：端到端 RTF ≈ 0.025–0.1（beam_size=1 贪心解码，比引擎层快约 2.5×），45s 中文 ~1.16s。
  服务 turbo 加载后稳态显存 ~2.3GB。
- **CPU int8**：端到端 RTF ≈ 0.12–0.28，45s 中文 ~7.0s。稳态 RSS ~1.38GB。

### 发现的 Bug（已修复）
- `app/main.py` 原 CPU 分支传 `device_index=None`，ctranslate2 要求整数 → CPU 模式加载即 500。
  已改为 `device_index=0`。修复后 CPU API 才能测出数据。

> 注意：API 层用 `beam_size=1`，引擎层用 `beam_size=5`，是不同解码配置，不可直接横比。

## 复跑

```bash
# 1. (可选) 重新生成音频
python3 benchmark/gen_audio.py
# 2. 跑 3 模式 (GPU 模式默认用 host GPU 0，可用 BENCH_GPU=1 切换)
bash benchmark/run_benchmark.sh
# 3. 生成报告
python3 benchmark/report.py   # -> benchmark/report.html
```

## 方法论

每条音频先 **1 次预热丢弃**（消除 cuDNN autotune / PTX JIT / 惰性显存分配的冷启动，
实测首帧可比热态慢 ~15×），再连续 **3 次** 取**中位数**为主数值（附最快/最慢），
显存取峰值。显存按 PID 经 NVML 统计该进程真实占用。GPU 模式运行在与线上服务共享的 GPU 0。
