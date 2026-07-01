#!/usr/bin/env python3
"""Benchmark faster-whisper large-v3-turbo across three modes.

Modes:
  fp16_gpu : device=cuda, compute_type=float16
  int8_gpu : device=cuda, compute_type=int8_float16   (int8 weights, fp16 compute)
  cpu      : device=cpu,  compute_type=int8

Methodology (per mode):
  1. Load model (timed) -> capture model footprint (GPU mem + peak RSS).
  2. Warmup: 1 throwaway transcription of a short clip (CUDA/cuDNN autotuning
     and lazy allocation settle on the first run).
  3. For each audio x 3 repeats: time transcription, track peak GPU memory
     (per-PID via NVML) and peak RSS. Report median time + min/max + peak mem.

Usage:
  python benchmark.py <mode> <audio_dir> <manifest.json> <out.json> [gpu_index]
"""
import gc
import glob
import json
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

REPEATS = 3
CPU_THREADS = 8           # mirror deployed config (WHISPER_THREADS=8)
BEAM_SIZE = 5
LANG_CODE = {"zh": "zh", "en": "en", "yue": "yue"}

MODES = {
    "fp16_gpu": dict(device="cuda", compute_type="float16"),
    "int8_gpu": dict(device="cuda", compute_type="int8_float16"),
    "cpu": dict(device="cpu", compute_type="int8"),
}


def rss_peak_mb() -> float:
    """Peak RSS of this process (high-water mark, kB -> MB). Linux ru_maxrss."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class GpuMonitor:
    """Poll per-process GPU memory via NVML; tracks a resettable peak."""

    def __init__(self):
        self.peak = 0.0
        self.cur = 0.0
        self._stop = False
        self._thread = None
        self.nvml = None
        self.handle = None

    def start(self):
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # no GPU / no nvidia-smi -> CPU mode
            print(f"  [GpuMonitor] disabled ({e})", flush=True)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        pid = os.getpid()
        while not self._stop:
            try:
                for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                    if p.pid == pid:
                        mb = p.usedGpuMemory / 1048576.0
                        if mb > self.peak:
                            self.peak = mb
                        self.cur = mb
                        break
            except Exception:
                pass
            time.sleep(0.005)

    def reset(self):
        self.peak = self.cur

    def stop(self):
        self._stop = True
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def resolve_model_path() -> str:
    pat = "/root/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*"
    cands = glob.glob(pat)
    if not cands:
        raise RuntimeError(f"turbo model not found under {pat}")
    return cands[0]


def transcribe_once(model, audio_path: str, lang: str):
    """Returns (elapsed_sec, text)."""
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio_path,
        language=LANG_CODE[lang],
        beam_size=BEAM_SIZE,
        vad_filter=False,
        word_timestamps=False,
    )
    text = " ".join(s.text for s in segments).strip()
    # faster-whisper transcribe is lazy -> force full run by consuming segments
    _ = len(text)
    elapsed = time.perf_counter() - t0
    return elapsed, text, info


def main():
    mode = sys.argv[1]
    audio_dir = Path(sys.argv[2])
    manifest = json.loads(Path(sys.argv[3]).read_text())
    out_json = Path(sys.argv[4])
    cfg = MODES[mode]

    print(f"=== benchmark mode={mode} device={cfg['device']} "
          f"compute={cfg['compute_type']} ===", flush=True)

    model_path = resolve_model_path()
    print(f"model: {model_path}", flush=True)

    mon = GpuMonitor()
    mon.start()

    # ---- load ----
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(
        model_path,
        device=cfg["device"],
        compute_type=cfg["compute_type"],
        cpu_threads=CPU_THREADS,
        num_workers=1,
    )
    load_time = time.perf_counter() - t0
    time.sleep(0.5)  # let monitor observe resident footprint
    gpu_after_load = mon.peak
    rss_after_load = rss_peak_mb()
    print(f"load_time={load_time:.3f}s  gpu_after_load={gpu_after_load:.0f}MB  "
          f"rss_after_load={rss_after_load:.0f}MB", flush=True)

    # ---- warmup (1 throwaway pass on shortest clip) ----
    warm = audio_dir / "zh_10s.wav"
    if warm.exists():
        print("warmup pass...", flush=True)
        mon.reset()
        transcribe_once(model, str(warm), "zh")
        print(f"  warmup done  gpu_peak={mon.peak:.0f}MB", flush=True)

    # ---- measured runs ----
    runs = []
    for item in manifest:
        path = audio_dir / item["file"]
        lang = item["lang"]
        dur = item["actual_sec"]

        times, gpu_peaks, text_preview = [], 0.0, ""
        for r in range(REPEATS):
            mon.reset()
            t0rss = rss_peak_mb()
            elapsed, text, info = transcribe_once(model, str(path), lang)
            times.append(elapsed)
            gpu_peaks = max(gpu_peaks, mon.peak)
            if r == 0:
                text_preview = text[:120]
            print(f"  {item['file']} run{r}: {elapsed:.3f}s "
                  f"gpu_peak={mon.peak:.0f}MB lang={info.language}", flush=True)

        runs.append(
            {
                "file": item["file"],
                "lang": lang,
                "lang_name": item["lang_name"],
                "duration_sec": dur,
                "infer_times_sec": [round(x, 4) for x in times],
                "infer_median_sec": round(statistics.median(times), 4),
                "infer_min_sec": round(min(times), 4),
                "infer_max_sec": round(max(times), 4),
                "rtf_median": round(statistics.median(times) / dur, 4),
                "gpu_peak_mb": round(gpu_peaks, 1),
                "rss_peak_mb": round(rss_peak_mb(), 1),
                "detected_language": lang,
                "text_preview": text_preview,
            }
        )

    mon.stop()

    result = {
        "mode": mode,
        "device": cfg["device"],
        "compute_type": cfg["compute_type"],
        "cpu_threads": CPU_THREADS,
        "beam_size": BEAM_SIZE,
        "repeats": REPEATS,
        "model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "load_time_sec": round(load_time, 3),
        "rss_after_load_mb": round(rss_after_load, 1),
        "gpu_after_load_mb": round(gpu_after_load, 1),
        "runs": runs,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_json}", flush=True)

    del model
    gc.collect()


if __name__ == "__main__":
    main()
