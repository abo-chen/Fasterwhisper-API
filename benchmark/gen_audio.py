#!/usr/bin/env python3
"""Generate TTS test audio via Fun-CosyVoice (localhost:9880) for the STT benchmark.

For each language (zh / en / yue) we synthesize one ~55s clip of duration-matched
text, then ffmpeg-trim it to exactly 10 / 25 / 45 seconds. Trimming (rather than
asking TTS for an exact length) gives clean, identical durations across languages
so RTF comparisons are fair.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

TTS_URL = "http://localhost:9880/v1/audio/speech"
VOICE = {"zh": "chinese", "en": "english", "yue": "cantonese"}
DURATIONS = [10, 25, 45]
OUT = Path(__file__).resolve().parent / "audio"
OUT.mkdir(parents=True, exist_ok=True)

# Duration-matched source text. Each is ~55s when read aloud so trimming to 45s
# stays inside real speech (no trailing silence).
TEXTS = {
    "zh": (
        "大家好，欢迎收听今天的节目。人工智能技术在过去几年里取得了飞速的发展，"
        "语音识别作为人机交互的重要入口，已经广泛应用于会议记录、字幕生成、"
        "智能客服和车载助手等众多场景。今天我们要测试的是一款基于大模型的语音识别系统，"
        "它能够在保证识别准确率的同时，大幅提升处理速度，从而满足实时应用的需求。"
        "接下来，我会播放几段不同时长的中文语音，用来评估系统在真实场景下的响应时间、"
        "资源占用以及稳定性表现，希望通过这次测试，能够为大家在选型和部署过程中"
        "提供有价值的参考依据，也欢迎大家持续关注我们的后续评测内容。"
    ),
    "en": (
        "Hello everyone, and welcome to today's presentation. "
        "Over the past few years, artificial intelligence has advanced at an extraordinary pace, "
        "and speech recognition has become a critical interface between humans and machines. "
        "It now powers meeting transcription, automatic subtitling, intelligent customer service, "
        "and in-car assistants across countless industries. "
        "Today we are evaluating a speech recognition system built on a large language model "
        "architecture, one that aims to deliver high accuracy while significantly improving "
        "processing speed to meet the demands of real-time applications. "
        "In the next few minutes, I will play several audio clips of varying lengths so that we "
        "can measure the system's response time, memory consumption, and overall stability under "
        "realistic conditions. We hope these results provide useful guidance for anyone selecting "
        "or deploying such a system in production environments. Thank you for listening."
    ),
    "yue": (
        "大家好，歡迎收聽今日嘅節目。人工智能技術喺過去幾年入面發展得好快，"
        "語音識別作為人機互動嘅重要入口，已經廣泛應用喺會議記錄、字幕生成、"
        "智能客服同埋車載助手咁多個場景。今日我哋要測試嘅，係一套基於大模型嘅語音識別系統，"
        "佢可以喺保證識別準確率嘅同時，大幅提升處理速度，從而滿足實時應用嘅需求。"
        "跟住落嚟，我會播放幾段唔同時長嘅廣東話語音，用嚟評估系統喺真實場景下嘅響應時間、"
        "資源佔用同埋穩定性表現，希望通過呢次測試，可以為大家喺選型同埋部署過程當中"
        "提供有價值嘅參考依據，亦都歡迎大家繼續留意我哋之後嘅評測內容。"
    ),
}

LANG_NAME = {"zh": "中文", "en": "English", "yue": "粤语"}


def synth(lang: str, path: Path) -> None:
    body = json.dumps(
        {
            "model": "fun-cosyvoice3",
            "input": TEXTS[lang],
            "voice": VOICE[lang],
            "response_format": "wav",
        }
    ).encode()
    req = urllib.request.Request(
        TTS_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        path.write_bytes(r.read())
    print(f"  [TTS] {lang} -> {path.name} ({path.stat().st_size} bytes)")


def ffprobe_dur(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    )
    return float(out.strip())


def ffprobe_sr(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of",
            "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    )
    return int(out.strip())


def trim(src: Path, dur: int, dst: Path) -> None:
    # Re-encode to canonical 16-bit PCM mono; keep source sample rate.
    sr = ffprobe_sr(src)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-t", str(dur), "-ac", "1", "-ar", str(sr),
            "-c:a", "pcm_s16le", str(dst),
        ]
    )


def main() -> None:
    manifest = []
    for lang in ("zh", "en", "yue"):
        full = OUT / f"{lang}_full.wav"
        print(f"[{LANG_NAME[lang]}] synthesizing full clip...")
        if not full.exists():
            synth(lang, full)
        full_dur = ffprobe_dur(full)
        sr = ffprobe_sr(full)
        print(f"  full clip duration={full_dur:.2f}s sr={sr}")
        if full_dur < max(DURATIONS) + 1:
            print(f"  !! WARNING: full clip {full_dur:.1f}s < {max(DURATIONS)}s target")
        for d in DURATIONS:
            dst = OUT / f"{lang}_{d}s.wav"
            trim(full, d, dst)
            actual = ffprobe_dur(dst)
            manifest.append(
                {
                    "file": dst.name,
                    "lang": lang,
                    "lang_name": LANG_NAME[lang],
                    "target_sec": d,
                    "actual_sec": round(actual, 3),
                    "sample_rate": sr,
                    "voice": VOICE[lang],
                }
            )
            print(f"  -> {dst.name} ({actual:.2f}s)")

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nmanifest written:", OUT / "manifest.json")
    print(f"generated {len(manifest)} clips")


if __name__ == "__main__":
    sys.exit(main())
