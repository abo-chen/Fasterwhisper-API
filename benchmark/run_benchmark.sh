#!/bin/bash
# Run the faster-whisper large-v3-turbo benchmark in 3 modes.
# GPU modes (fp16_gpu, int8_gpu) -> dedicated container pinned to host GPU 0.
# CPU mode -> container with no GPU visible.
set -uo pipefail

DIR="/home/abel-chen/.local/opt/fasterwhisper/benchmark"
GPU="${BENCH_GPU:-0}"
MODELS_VOL="/home/abel-chen/.local/opt/fasterwhisper/whisper-models:/root/.cache/huggingface:ro"

run_mode () {
  local mode="$1"; shift
  echo "============================================================"
  echo "  MODE = $mode   (extra: $*)"
  echo "============================================================"
  docker run --rm --name "bench-$mode" "$@" \
    -v "$DIR:/bench" \
    -v "$MODELS_VOL" \
    -e HF_HUB_OFFLINE=1 \
    whisper-bench:latest python /bench/benchmark.py "$mode" \
      /bench/audio /bench/audio/manifest.json "/bench/results/$mode.json" \
    2>&1 | tee "$DIR/logs/$mode.log"
}

# GPU modes share host GPU 0 (documented in report). int8_gpu runs second so the
# fp16 autotuning state from the first does not bias the int8 numbers (fresh proc).
run_mode fp16_gpu --gpus "\"device=$GPU\""
run_mode int8_gpu --gpus "\"device=$GPU\""
# CPU mode: force no CUDA.
run_mode cpu -e CUDA_VISIBLE_DEVICES=""

echo "============================================================"
echo "All modes done. Results:"
ls -la "$DIR/results/"
