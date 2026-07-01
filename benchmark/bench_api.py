#!/usr/bin/env python3
"""API-layer benchmark: call the running Faster-Whisper HTTP service and time
end-to-end transcription (upload -> result). Warmup 1x, then 3x per clip.

Usage: python bench_api.py <base_url> <model> <manifest.json> <out.json> <label> [lang...]
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPEATS = 3
LANG_CODE = {"zh": "zh", "en": "en", "yue": "yue"}


def post_transcribe(base_url, model, wav_path, lang):
    boundary = uuid.uuid4().hex
    fields = {"model": model, "language": LANG_CODE[lang], "response_format": "json"}
    body = b""
    for k, v in fields.items():
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
        ).encode()
    fn = Path(wav_path).name
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f'filename="{fn}"\r\nContent-Type: audio/wav\r\n\r\n'
    ).encode()
    body += Path(wav_path).read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return time.perf_counter() - t0, data.get("text", "")


def main():
    base_url, model, manifest_path, out_json, label = sys.argv[1:6]
    langs = sys.argv[6:] or ["zh", "en", "yue"]
    manifest = json.loads(Path(manifest_path).read_text())
    manifest = [m for m in manifest if m["lang"] in langs]

    # warmup (also ensures model is loaded)
    warm = next(m for m in manifest if m["lang"] == "zh") if "zh" in langs else manifest[0]
    print(f"[{label}] warmup {warm['file']} ...", flush=True)
    t, _ = post_transcribe(base_url, model, str(Path(manifest_path).parent / warm["file"]), warm["lang"])
    print(f"  warmup {t:.3f}s", flush=True)

    runs = []
    for item in manifest:
        wav = str(Path(manifest_path).parent / item["file"])
        times, text_preview = [], ""
        for r in range(REPEATS):
            t, text = post_transcribe(base_url, model, wav, item["lang"])
            times.append(t)
            if r == 0:
                text_preview = text[:120]
            print(f"  {item['file']} run{r}: {t:.3f}s", flush=True)
        runs.append(
            {
                "file": item["file"],
                "lang": item["lang"],
                "lang_name": item["lang_name"],
                "duration_sec": item["actual_sec"],
                "times_sec": [round(x, 4) for x in times],
                "median_sec": round(statistics.median(times), 4),
                "min_sec": round(min(times), 4),
                "max_sec": round(max(times), 4),
                "rtf": round(statistics.median(times) / item["actual_sec"], 4),
                "text_preview": text_preview,
            }
        )

    result = {
        "label": label,
        "base_url": base_url,
        "model": model,
        "repeats": REPEATS,
        "runs": runs,
    }
    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
