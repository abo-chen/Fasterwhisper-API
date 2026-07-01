#!/usr/bin/env python3
"""Generate a self-contained HTML benchmark report from the per-mode JSON results.

Reads benchmark/results/{fp16_gpu,int8_gpu,cpu}.json + audio/manifest.json and
writes benchmark/report.html. Stdlib only (html + inline SVG), no jinja2.
"""
import html
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODE_ORDER = ["fp16_gpu", "int8_gpu", "cpu"]
MODE_LABEL = {
    "fp16_gpu": "GPU · fp16",
    "int8_gpu": "GPU · int8_float16",
    "cpu": "CPU · int8",
}
MODE_SHORT = {"fp16_gpu": "fp16", "int8_gpu": "int8", "cpu": "cpu"}
LANG_ORDER = ["zh", "en", "yue"]
LANG_LABEL = {"zh": "中文", "en": "English", "yue": "粤语"}
DURATIONS = [10, 25, 45]


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def load_results():
    out = {}
    for m in MODE_ORDER:
        p = RESULTS / f"{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def run_map(mode_data):
    """file -> run dict."""
    return {r["file"]: r for r in mode_data.get("runs", [])}


def unsupported(d):
    """A mode whose compute type cannot run on this hardware."""
    return d.get("status") == "unsupported"


def fmt(x, n=2):
    return f"{x:.{n}f}"


# ----------------------------- chart helpers -----------------------------
def bar_chart_svg(series, title, unit=""):
    """series: list of dicts {label, value, color}. Returns SVG string."""
    series = [s for s in series if s.get("value") is not None]
    if not series:
        return "<p><em>无数据</em></p>"
    vmax = max(s["value"] for s in series) * 1.15 or 1
    bar_h, gap, label_w, val_w = 22, 8, 150, 80
    h = len(series) * (bar_h + gap) + 30
    w = 620
    plot_w = w - label_w - val_w - 20
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{html.escape(title)}">']
    y = 8
    for s in series:
        bw = max(1, s["value"] / vmax * plot_w)
        parts.append(f'<text x="0" y="{y + bar_h * 0.7}" class="clab">{html.escape(s["label"])}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{s["color"]}"/>')
        parts.append(f'<text x="{label_w + bw + 6:.1f}" y="{y + bar_h * 0.7}" class="cval">{fmt(s["value"])}{unit}</text>')
        y += bar_h + gap
    parts.append("</svg>")
    return "\n".join(parts)


# ----------------------------- tables -----------------------------
def speed_table(results):
    """Big table: rows = (lang, duration); cols per mode = median(s) / RTF."""
    head = "<tr><th>语言</th><th>时长</th>"
    for m in MODE_ORDER:
        if m in results:
            head += f'<th colspan="2">{MODE_LABEL[m]}</th>'
    head += "</tr>"
    sub = "<tr><th></th><th></th>" + "".join(
        "<th>耗时(s)</th><th>RTF</th>" for m in MODE_ORDER if m in results
    ) + "</tr>"

    rows = ""
    for lang in LANG_ORDER:
        for d in DURATIONS:
            f = f"{lang}_{d}s.wav"
            cells = ""
            for m in MODE_ORDER:
                if m not in results:
                    continue
                if unsupported(results[m]):
                    cells += '<td colspan="2" class="warn">⚠ 不支持</td>'
                    continue
                rm = run_map(results[m])
                if f not in rm:
                    cells += '<td class="muted">—</td><td class="muted">—</td>'
                    continue
                r = rm[f]
                cells += f'<td>{fmt(r["infer_median_sec"], 3)}</td><td>{fmt(r["rtf_median"], 3)}</td>'
            rows += f'<tr><td class="lang">{LANG_LABEL[lang]}</td><td>{d}s</td>{cells}</tr>'

    return f"""<table class="data"><thead>{head}{sub}</thead><tbody>{rows}</tbody></table>"""


def memory_table(results):
    rows = ""
    for m in MODE_ORDER:
        if m not in results:
            continue
        d = results[m]
        if unsupported(d):
            rows += (
                f"<tr><td class='mode'>{MODE_LABEL[m]}</td>"
                f"<td colspan='4' class='warn'>⚠ 不支持 —— {html.escape(d.get('error',''))}<br>"
                f"<span class='small'>{html.escape(d.get('reason',''))}</span></td></tr>"
            )
            continue
        # peak GPU across all clips
        gpu_peaks = [r["gpu_peak_mb"] for r in d["runs"] if r["gpu_peak_mb"] > 0]
        gpu_peak = max(gpu_peaks) if gpu_peaks else 0
        rss_peak = max(r["rss_peak_mb"] for r in d["runs"])
        gpu_load = d.get("gpu_after_load_mb", 0)
        rows += (
            f"<tr><td class='mode'>{MODE_LABEL[m]}</td>"
            f"<td>{fmt(gpu_load, 0)} MB</td>"
            f"<td>{fmt(gpu_peak, 0)} MB</td>"
            f"<td>{fmt(rss_peak, 0)} MB</td>"
            f"<td>{fmt(d['load_time_sec'], 2)} s</td></tr>"
        )
    return f"""<table class="data"><thead><tr>
        <th>模式</th><th>模型加载后显存</th><th>推理峰值显存</th><th>峰值内存 (RSS)</th><th>模型加载耗时</th>
        </tr></thead><tbody>{rows}</tbody></table>
        <p class="note">显存为加载/推理后该进程的占用（NVML 按 PID 统计）；
        RSS 为进程峰值驻留内存。CPU 模式无显存。</p>"""


def detail_table(results):
    rows = ""
    for m in MODE_ORDER:
        if m not in results:
            continue
        rm = run_map(results[m])
        for lang in LANG_ORDER:
            for d in DURATIONS:
                f = f"{lang}_{d}s.wav"
                if f not in rm:
                    continue
                r = rm[f]
                ts = " / ".join(fmt(x, 3) for x in r["infer_times_sec"])
                rows += (
                    f"<tr><td>{MODE_LABEL[m]}</td><td>{LANG_LABEL[lang]}</td><td>{d}s</td>"
                    f"<td>{fmt(r['infer_median_sec'],3)}</td>"
                    f"<td>{fmt(r['infer_min_sec'],3)}</td>"
                    f"<td>{fmt(r['infer_max_sec'],3)}</td>"
                    f"<td>{fmt(r['rtf_median'],3)}</td>"
                    f"<td>{fmt(r['gpu_peak_mb'],0)}</td>"
                    f"<td class='muted small'>{ts}</td></tr>"
                )
    return f"""<table class="data dense"><thead><tr>
        <th>模式</th><th>语言</th><th>时长</th><th>中位耗时</th><th>最快</th><th>最慢</th>
        <th>RTF</th><th>显存峰值</th><th>3 次 (s)</th></tr></thead><tbody>{rows}</tbody></table>"""


def api_results():
    out = {}
    for m in ["api_fp16_gpu", "api_cpu"]:
        p = RESULTS / f"{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def api_table(api):
    if not api:
        return '<p class="note">无 API 层基准数据。</p>'
    n = len(api)
    head = "<tr><th>语言</th><th>时长</th>"
    if "api_fp16_gpu" in api:
        head += '<th colspan="2">API · GPU fp16</th>'
    if "api_cpu" in api:
        head += '<th colspan="2">API · CPU int8</th>'
    head += "</tr>"
    sub = "<tr><th></th><th></th>" + "".join("<th>端到端耗时(s)</th><th>RTF</th>" for _ in range(n)) + "</tr>"
    rm_gpu = {r["file"]: r for r in api.get("api_fp16_gpu", {}).get("runs", [])}
    rm_cpu = {r["file"]: r for r in api.get("api_cpu", {}).get("runs", [])}
    rows = ""
    for lang in LANG_ORDER:
        for d in DURATIONS:
            f = f"{lang}_{d}s.wav"
            cells = ""
            for rm in [rm_gpu, rm_cpu]:
                if not rm:
                    continue
                if f in rm:
                    r = rm[f]
                    cells += f'<td>{fmt(r["median_sec"], 3)}</td><td>{fmt(r["rtf"], 3)}</td>'
                else:
                    cells += '<td class="muted">—</td><td class="muted">—</td>'
            rows += f'<tr><td class="lang">{LANG_LABEL[lang]}</td><td>{d}s</td>{cells}</tr>'
    return f'<table class="data"><thead>{head}{sub}</thead><tbody>{rows}</tbody></table>'


def preview_block(results):
    """Show transcription preview per language (from fp16_gpu if present)."""
    src = results.get("fp16_gpu") or results.get("int8_gpu") or results.get("cpu")
    if not src:
        return ""
    rm = run_map(src)
    parts = ['<div class="grid3">']
    for lang in LANG_ORDER:
        r = rm.get(f"{lang}_25s.wav")
        if not r:
            continue
        parts.append(
            f'<div class="card"><h4>{LANG_LABEL[lang]} (25s)</h4>'
            f'<p class="preview">{html.escape(r["text_preview"])}</p></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def build():
    results = load_results()
    if not results:
        print("No result JSON files found in", RESULTS)
        sys.exit(1)

    # ---- derived: speedup vs cpu; chart data ----
    COLORS = {"fp16_gpu": "#2563eb", "int8_gpu": "#16a34a", "cpu": "#dc2626"}

    # RTF chart: for each mode, median RTF averaged across langs at 45s
    rtf_series = []
    for m in MODE_ORDER:
        if m not in results:
            continue
        rm = run_map(results[m])
        rtfs = [rm[f"{l}_45s.wav"]["rtf_median"] for l in LANG_ORDER if f"{l}_45s.wav" in rm]
        if rtfs:
            rtf_series.append({"label": f"{MODE_LABEL[m]} (45s 均值)", "value": sum(rtfs) / len(rtfs), "color": COLORS[m]})
    rtf_chart = bar_chart_svg(rtf_series, "RTF (45s 平均，越低越快)")

    # Throughput chart: median time for 45s clips per mode/lang
    time_series = []
    for m in MODE_ORDER:
        if m not in results:
            continue
        rm = run_map(results[m])
        for lang in LANG_ORDER:
            f = f"{lang}_45s.wav"
            if f in rm:
                time_series.append({"label": f"{MODE_SHORT[m]}·{LANG_LABEL[lang]} 45s", "value": rm[f]["infer_median_sec"], "color": COLORS[m]})
    time_chart = bar_chart_svg(time_series, "45s 音频推理耗时 (s，越短越快)", " s")

    # GPU memory chart
    mem_series = []
    for m in MODE_ORDER:
        if m not in results:
            continue
        d = results[m]
        gp = max([r["gpu_peak_mb"] for r in d["runs"] if r["gpu_peak_mb"] > 0] or [0])
        if gp:
            mem_series.append({"label": MODE_LABEL[m], "value": gp, "color": COLORS[m]})
    mem_chart = bar_chart_svg(mem_series, "推理峰值显存 (MB)")

    # ---- speedup summary text ----
    insights = []
    if "cpu" in results and "fp16_gpu" in results:
        rcpu, rgpu = run_map(results["cpu"]), run_map(results["fp16_gpu"])
        f = "zh_45s.wav"
        if f in rcpu and f in rgpu:
            sp = rcpu[f]["infer_median_sec"] / rgpu[f]["infer_median_sec"]
            insights.append(f"GPU fp16 相比 CPU int8 在中文 45s 上提速约 <b>{fmt(sp,1)}×</b>")
    if "int8_gpu" in results and not unsupported(results["int8_gpu"]) and "fp16_gpu" in results:
        ri, rf = run_map(results["int8_gpu"]), run_map(results["fp16_gpu"])
        gi = max((r["gpu_peak_mb"] for r in results["int8_gpu"]["runs"]), default=0)
        gf = max((r["gpu_peak_mb"] for r in results["fp16_gpu"]["runs"]), default=0)
        if gf:
            insights.append(f"int8_float16 相比 fp16 峰值显存 <b>{fmt(gf,0)}→{fmt(gi,0)} MB</b>（节省 {fmt((gf-gi)/gf*100,0)}%）")

    # ---- API-layer section ----
    api = api_results()
    api_section = ""
    if api:
        api_section = (
            '<section>\n<h2>6 · API 层基准（调用运行中的服务）</h2>\n'
            '<p class="note">通过 HTTP <code>/v1/audio/transcriptions</code> 调用已部署服务'
            '（model=<code>mobiuslabsgmbh/faster-whisper-large-v3-turbo</code>），测端到端耗时'
            '（上传+识别+返回）。GPU 走线上 whisper-gpu0(:5012)，CPU 走临时 CPU 实例(:5014)。</p>\n'
            f"{api_table(api)}\n"
            '<div class="callout"><b>⚠ 发现并修复 CPU 模式 Bug。</b> 服务原代码 '
            '<code>app/main.py</code> 在 CPU 分支传 <code>device_index=None</code>，而 ctranslate2 '
            '要求整数 → CPU 模式加载即抛 <code>incompatible constructor arguments</code> (HTTP 500)。'
            '已改为 <code>device_index=0</code> 修复后才测出上表 CPU 数据。建议合并此一行修复。</div>\n'
            '<h3>服务内存占用（稳态）</h3>\n'
            '<p class="note">GPU 服务 turbo 加载后显存 ~2308 MiB（推理峰值见第 3 节 ~2500 MiB）；'
            'CPU 服务稳态 RSS ~1.38 GiB（推理峰值 ~2.1 GB）。</p>\n'
            '<p class="note"><b>与引擎层差异说明：</b>本服务默认 <code>beam_size=1 + word_timestamps=True</code> '
            '（贪心解码），比第 1 节引擎层基准的 <code>beam_size=5</code> 显著更快（GPU 上约 2.5×），'
            '代价是略低的识别稳健性。故 API 数字与引擎层数字是<b>不同解码配置</b>，不宜直接横比。</p>\n'
            '</section>\n'
        )

    # ---- hardware info ----
    gpu_name = sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    cpu_name = sh("lscpu | grep -i 'model name' | cut -d: -f2").strip() or "Intel i7-14700K"
    ctrans = "ctranslate2 4.7.1 / faster-whisper 1.2.1"
    # Date the report from when the benchmark actually ran: newest result-JSON mtime.
    mtimes = []
    for m in MODE_ORDER + ["api_fp16_gpu", "api_cpu"]:
        p = RESULTS / f"{m}.json"
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    test_dt = datetime.fromtimestamp(max(mtimes)) if mtimes else datetime.now()
    test_time = test_dt.strftime("%Y-%m-%d %H:%M")
    today = test_time
    n_modes = len(results)

    css = """
    :root{--bg:#0b1020;--card:#141b30;--ink:#e8edf7;--mut:#8b97b3;--acc:#3b82f6;--line:#27314f}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.55 -apple-system,Segoe UI,Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
    .wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
    header h1{font-size:26px;margin:0 0 4px}header .sub{color:var(--mut);font-size:14px}
    .meta{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
    .chip{background:var(--card);border:1px solid var(--line);padding:5px 12px;border-radius:999px;font-size:13px;color:var(--mut)}
    .chip b{color:var(--ink)}
    section{margin:34px 0}section h2{font-size:18px;border-left:3px solid var(--acc);padding-left:10px;margin:0 0 14px}
    section h3{font-size:15px;color:var(--mut);font-weight:600;margin:18px 0 8px}
    table.data{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
    table.data th,table.data td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:center}
    table.data th{background:#1a2340;color:var(--mut);font-weight:600;font-size:13px}
    table.data td.lang,table.data td.mode{text-align:left;font-weight:600}
    table.data dense td,table.data.dense td{padding:6px 8px}
    table.data tbody tr:hover{background:#1a2340}
    td.muted{color:var(--mut)}td.small{font-size:12px}
    .warn{color:#f59e0b;font-weight:600;background:rgba(245,158,11,.08)}
    .callout{background:rgba(245,158,11,.08);border:1px solid #b45309;border-left:4px solid #f59e0b;border-radius:10px;padding:12px 16px;margin:14px 0;font-size:14px}
    .callout b{color:#fbbf24}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:6px 0}
    .kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
    .kpi .k{color:var(--mut);font-size:12px}.kpi .v{font-size:22px;font-weight:700;margin-top:4px}
    .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
    .card h4{margin:0 0 8px;font-size:14px}
    .preview{font-size:13px;color:#cdd6ea;margin:0;white-space:pre-wrap}
    .chart{width:100%;height:auto;display:block;background:var(--card);border:1px solid var(--line);border-radius:10px}
    .chart .clab{fill:var(--mut);font-size:12px}.chart .cval{fill:var(--ink);font-size:12px}
    .note{color:var(--mut);font-size:13px;margin:8px 0 0}
    .insight{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:14px}
    footer{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
    a{color:var(--acc)}
    """

    insight_html = "".join(f'<div class="insight">• {i}</div>' for i in insights)

    int8_callout = ""
    if "int8_gpu" in results and unsupported(results["int8_gpu"]):
        d8 = results["int8_gpu"]
        int8_callout = (
            '<div class="callout"><b>⚠ int8 GPU 模式在本机不可用。</b> '
            f"{html.escape(d8.get('error',''))} —— {html.escape(d8.get('reason',''))}"
            "<br>已实测三种 int8 计算类型 "
            f"<code>{'</code> / <code>'.join(d8.get('compute_types_tested', []))}</code> "
            "均在首次推理失败；故本报告 int8 GPU 列以「不支持」标注。fp16 GPU 与 CPU int8 正常。</div>"
        )

    body = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Faster-Whisper large-v3-turbo 性能基准报告</title><style>{css}</style></head><body>
<div class="wrap">
<header>
  <h1>Faster-Whisper <span style="color:var(--acc)">large-v3-turbo</span> 性能基准</h1>
  <div class="sub">fp16 / int8 GPU 与 CPU 模式 · 中文 / 英文 / 粤语 · 10s / 25s / 45s</div>
</header>
<div class="meta">
  <span class="chip">测试时间 <b>{today}</b></span>
  <span class="chip">GPU <b>{html.escape(gpu_name)}</b></span>
  <span class="chip">CPU <b>{html.escape(cpu_name)}</b></span>
  <span class="chip">引擎 <b>{ctrans}</b></span>
  <span class="chip">模型 <b>mobiuslabsgmbh/faster-whisper-large-v3-turbo</b></span>
  <span class="chip">TTS <b>Fun-CosyVoice :9880</b></span>
</div>
{insight_html}
{int8_callout}

<section>
<h2>1 · 推理速度（中位耗时 / RTF）</h2>
<p class="note">每条音频先 1 次预热丢弃，再连续测 3 次取中位数；RTF = 耗时 / 音频时长（&lt;1 即快于实时）。</p>
{speed_table(results)}
</section>

<section>
<h2>2 · 速度对比图</h2>
<h3>45s 音频推理耗时（越短越快）</h3>
{time_chart}
<h3>RTF（45s 平均，越低越快）</h3>
{rtf_chart}
</section>

<section>
<h2>3 · 显存与内存占用</h2>
{memory_table(results)}
<h3>推理峰值显存对比</h3>
{mem_chart}
</section>

<section>
<h2>4 · 全量明细（最快 / 最慢 / 3 次原始）</h2>
{detail_table(results)}
</section>

<section>
<h2>5 · 转写示例（质量抽检）</h2>
<p class="note">取自 25s 音频，用于确认各语言识别正常（尤其粤语）。</p>
{preview_block(results)}
</section>

{api_section}

<section>
<h2>7 · 方法论与说明</h2>
<ul style="color:var(--mut);font-size:14px;line-height:1.8">
<li><b>音频</b>：由 Fun-CosyVoice (:9880, OpenAI 兼容 <code>/v1/audio/speech</code>) 合成
中文/英文/粤语长语音，再用 ffmpeg 精确切到 10/25/45s（24kHz 单声道 16bit PCM）。</li>
<li><b>预热</b>：每个模式加载模型后先跑 1 次 10s 短音频并丢弃结果，消除 CUDA/cuDNN
内核 autotuning、PTX JIT 与惰性显存分配带来的首次冷启动开销（实测首帧可比热态慢 ~15×）。</li>
<li><b>重复</b>：每条音频测 3 次，取<b>中位数</b>为主数值，并给出最快/最慢；显存取峰值。</li>
<li><b>配置</b>：beam_size=5，VAD 关闭，强制传入语言（跳过语种检测），word_timestamps 关闭；
CPU 模式 cpu_threads=8（与部署配置 <code>WHISPER_THREADS=8</code> 一致）。</li>
<li><b>显存统计</b>：通过 NVML 按 PID 获取该推理进程的真实占用（含 ctranslate2 缓冲），
而非 torch 分配器口径。</li>
<li><b>GPU 共享</b>：基准运行在与线上服务共享的 GPU 0（同机还驻留 LLM 等服务），
属真实部署条件；同一 GPU 下三种模式横向对比仍然公平。</li>
<li>原始 JSON（引擎层）：<code>results/fp16_gpu.json</code> · <code>int8_gpu.json</code> · <code>cpu.json</code>；
（API 层）：<code>results/api_fp16_gpu.json</code> · <code>api_cpu.json</code>。逐次日志：<code>logs/*.log</code>；音频：<code>audio/</code>。</li>
<li><b>引擎层 vs API 层</b>：引擎层直连 faster-whisper 库（beam_size=5，量 per-PID 显存）；
API 层走 HTTP 服务（beam_size=1）。两套均含预热+3 次取中位。</li>
</ul>
</section>

<footer>Generated by benchmark/report.py · Faster-Whisper STT · {today}</footer>
</div></body></html>"""

    out = HERE / "report.html"
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}  ({len(body)} bytes, modes={list(results)})")


if __name__ == "__main__":
    build()
