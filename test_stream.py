#!/usr/bin/env python3
"""测试 gpt_streaming（语义流式）真的生效了吗

两种模式，默认第一种：

  直接模式（不用起服务，不用 curl）
      .\\.venv\\Scripts\\python.exe test_stream.py --ref D:\\ref.wav

  HTTP 模式（需要另开终端跑 api_v2.py）
      .\\.venv\\Scripts\\python.exe test_stream.py --http --ref D:\\ref.wav

它做的事
--------
1. 调 TTS.run()，逐块计时
2. 打印每块到达时刻 / 样本数 / 折算音频秒
3. 拼成 wav 存下来
4. 给判定：真流式 / 疑似攒批
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BAR = "=" * 70


def head(t: str) -> None:
    print()
    print(BAR)
    print(t)
    print(BAR)


def write_wav(path: Path, sr: int, arrays: list) -> None:
    import numpy as np

    if not arrays:
        return
    data = np.concatenate([a for a in arrays if a is not None and a.size])
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def report(marks: list, sizes: list, sr: int, t0: float, out: Path, mode: str) -> int:
    head("计时明细")
    if not marks:
        print("  一个块都没收到。")
        return 1

    print(f"  {'#':>3}  {'到达':>9}  {'间隔':>7}  {'样本':>8}  {'音频秒':>7}")
    print("  " + "-" * 46)
    prev = None
    for i, (t, n) in enumerate(zip(marks, sizes), 1):
        gap = "" if prev is None else f"{t - prev:.2f}"
        print(f"  {i:>3}  {t - t0:>8.2f}s  {gap:>6}s  {n:>8}  {n / sr:>6.2f}")
        prev = t

    first = marks[0] - t0
    total = marks[-1] - t0
    audio_sec = sum(sizes) / sr

    head("汇总")
    print(f"  模式      : {mode}")
    print(f"  块数      : {len(marks)}")
    print(f"  首块到达  : {first:.2f}s")
    print(f"  全部结束  : {total:.2f}s")
    print(f"  音频总长  : {audio_sec:.2f}s")
    if total > 0:
        print(f"  RTF       : {audio_sec / total:.2f}  (>1 比实时快)")
    print(f"  输出      : {out}")

    head("判定")
    n = len(marks)
    if n == 1:
        print("  [X] 只收到 1 块 —— 没有流式效果，整句算完才吐。")
        print("      可能原因：")
        print("        · 模型不是 v2ProPlus / v5turbo / v5dev")
        print("        · gpt_streaming 没传或没生效")
        print("        · streaming_mode 不是 2/3")
        return 1

    ratio = first / total if total > 0 else 1.0
    print(f"  首块/总时长 = {ratio:.2f}")
    if n >= 3 and ratio < 0.70:
        print("  [OK] 真流式：首块明显早于总时长，且分了多块。")
        ok = True
    elif n >= 2 and ratio < 0.85:
        print("  [~] 部分流式：有分块但首块偏晚。可试更小的 min_chunk_length")
        print("      或更细的 text_split_method（如 cut1）。")
        ok = True
    else:
        print("  [X] 疑似攒批：分了块但首块几乎等到最后才来。")
        ok = False

    print()
    print("  注：首次运行含模型加载，首块延迟偏高。再跑一次才是干净数字。")
    return 0 if ok else 1



# ============================================================ 找参考音频

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _duration(path: Path) -> float | None:
    """尽量读时长；读不到返回 None。"""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return None
    return None


def scan_refs(root: Path, limit: int = 25) -> list[tuple[Path, float | None]]:
    """在目录里找 1.5~15 秒的音频，当参考音频的候选。"""
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if len(out) >= limit * 4:
            break
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            continue
        if any(seg in p.parts for seg in (".venv", "__pycache__", ".git", "logs")):
            continue
        d = _duration(p)
        if d is not None and not (1.5 <= d <= 15.0):
            continue
        out.append((p, d))
        if len(out) >= limit:
            break
    return out


def print_candidates(cands: list[tuple[Path, float | None]], root: Path) -> None:
    if not cands:
        print(f"  在 {root} 下没找到合适的音频（1.5~15 秒）。")
        print("  换个目录扫：--scan D:\\你的语音目录")
        return
    print(f"  在 {root} 下找到 {len(cands)} 个候选：")
    print(f"  {'#':>2}  {'时长':>7}  路径")
    print("  " + "-" * 62)
    for i, (path, d) in enumerate(cands, 1):
        ds = f"{d:.2f}s" if d is not None else "  ?  "
        print(f"  {i:>2}  {ds:>7}  {path}")
    print()
    print("  直接把其中一条路径传给 --ref 即可，例如：")
    print(f'    .\\.venv\\Scripts\\python.exe test_stream.py --ref "{cands[0][0]}"')


# ============================================================ 直接模式

def run_direct(args) -> int:
    head("直接调用 TTS pipeline")
    from copy import deepcopy

    import numpy as np

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    yaml_path = "GPT_SoVITS/configs/tts_infer.yaml"

    if args.version:
        if args.version not in TTS_Config.default_configs:
            print(f"  未知版本 {args.version}")
            print(f"  可选：{list(TTS_Config.default_configs)}")
            return 1
        cfg_dict = deepcopy(TTS_Config.default_configs[args.version])
        cfg_dict["device"] = "xpu"
        cfg_dict["is_half"] = False
        cfg_payload = {"custom": cfg_dict}
        print(f"  版本：{args.version}（来自 default_configs 模板）")
    else:
        cfg_payload = yaml_path
        print(f"  版本：读 {yaml_path} 的 custom 段")

    tts = TTS(cfg_payload)
    cfg = tts.configs
    print(f"  device : {cfg.device}")
    print(f"  version: {cfg.version}")

    for label, attr in (("GPT(t2s)", "t2s_weights_path"), ("SoVITS", "vits_weights_path")):
        p = Path(getattr(cfg, attr, ""))
        ok = p.exists()
        print(f"  {label:<10}: {p}  {'OK' if ok else '缺失!'}")
        if not ok:
            print()
            print("  权重缺了，先跑：python download_models.py")
            return 1

    req = {
        "text": args.text,
        "text_lang": "zh",
        "ref_audio_path": str(Path(args.ref).resolve()),
        "prompt_text": args.prompt_text or "",
        "prompt_lang": "zh",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "split_bucket": False,
        "parallel_infer": False,
        "repetition_penalty": 1.35,
        "streaming_mode": args.mode,
        "gpt_streaming": True,
        "min_chunk_length": 16,
        "fragment_interval": 0.0,
    }

    print()
    print(f"  gpt_streaming={req['gpt_streaming']}  streaming_mode={req['streaming_mode']}")
    print(f"  文本：{args.text}")
    print()

    marks, sizes, arrays = [], [], []
    sr_seen = 32000
    t0 = time.perf_counter()
    try:
        for item in tts.run(req):
            now = time.perf_counter()
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                print(f"  [skip] 非 (sr, audio)：{type(item)}")
                continue
            sr, audio = item
            sr_seen = int(sr)
            arr = np.asarray(audio).reshape(-1)
            marks.append(now)
            sizes.append(int(arr.size))
            arrays.append(arr)
            silent = "静音块" if arr.size and not arr.any() else "有声"
            print(f"  +{now - t0:5.2f}s  #{len(marks):<2} {arr.size:>6} 样本 "
                  f"= {arr.size / sr:4.2f}s  {silent}")
    except KeyboardInterrupt:
        print("\n  手动中断")
    except Exception as e:
        print()
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(limit=8)
        return 1

    if not marks:
        print("  没拿到任何音频块")
        return 1

    out = Path(args.out)
    write_wav(out, sr_seen, arrays)
    return report(marks, sizes, sr_seen, t0, out, f"direct / mode={args.mode}")


# ============================================================ HTTP 模式

def run_http(args) -> int:
    head("通过 HTTP API")
    try:
        import requests
    except ImportError:
        print("  需要 requests：pip install requests")
        return 1

    body = {
        "text": args.text,
        "text_lang": "zh",
        "ref_audio_path": str(Path(args.ref).resolve()),
        "prompt_text": args.prompt_text or "",
        "prompt_lang": "zh",
        "text_split_method": "cut5",
        "batch_size": 1,
        "gpt_streaming": True,
        "streaming_mode": args.mode,
        "media_type": "raw",
    }
    print(f"  POST {args.api}")

    marks, sizes = [], []
    raw = bytearray()
    t0 = time.perf_counter()
    try:
        with requests.post(args.api, json=body, stream=True, timeout=600) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {r.text[:500]}")
                return 1
            for block in r.iter_content(chunk_size=4096):
                if not block:
                    continue
                now = time.perf_counter()
                raw.extend(block)
                marks.append(now)
                sizes.append(len(block) // 2)     # int16
                print(f"  +{now - t0:5.2f}s  {len(block):>6} bytes")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return 1

    out = Path(args.out)
    out.write_bytes(bytes(raw))
    print(f"  原始字节已写入 {out}（media_type=raw）")
    return report(marks, sizes, 32000, t0, out, f"http / mode={args.mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="测 gpt_streaming 是否真流式")
    ap.add_argument("--ref", default="", help="参考音频路径（wav/flac/mp3）")
    ap.add_argument("--scan", default="", help="只扫描某个目录下的音频候选，不测试")
    ap.add_argument("--text", default="这一句会边想边说，我们看看是不是真的分块出来。")
    ap.add_argument("--prompt-text", default="", help="参考音频对应文本（可空）")
    ap.add_argument("--version", default="v2ProPlus",
                    help="模型版本，默认 v2ProPlus")
    ap.add_argument("--mode", type=int, default=3, choices=[2, 3])
    ap.add_argument("--http", action="store_true", help="走 HTTP API")
    ap.add_argument("--api", default="http://127.0.0.1:9880/tts")
    ap.add_argument("--out", default="out_stream.wav")
    args = ap.parse_args()

    # ---- 只扫描不测试
    if args.scan:
        root = Path(args.scan)
        head(f"扫描参考音频候选：{root}")
        print_candidates(scan_refs(root), root)
        return 0

    ref = Path(args.ref).expanduser() if args.ref.strip() else None
    if ref is not None and not ref.is_absolute():
        ref = (ROOT / ref).resolve()
    if ref is None or not ref.exists():
        print()
        print(f"[ABORT] 参考音频不存在：{ref if ref else '(未提供 --ref)'}")
        print()
        print("  --ref 要给一个真实存在的音频文件（wav/flac/mp3），不是占位符。")
        print("  需要一段 3~10 秒的干净人声。")
        print()
        head("自动找一下（扫描仓库目录）")
        cands = scan_refs(ROOT, limit=15)
        print_candidates(cands, ROOT)
        print()
        print("  如果没有合适的，指定自己的目录再扫一次：")
        print('    .\\.venv\\Scripts\\python.exe test_stream.py --scan "D:\\你的语音目录"')
        return 1

    d = _duration(ref)
    print("gpt_streaming 测试")
    print(f"参考音频：{ref}" + (f"  ({d:.2f}s)" if d else ""))
    if d is not None and d < 1.0:
        print("  [WARN] 音频不足 1 秒，参考效果会很差")

    return run_http(args) if args.http else run_direct(args)


if __name__ == "__main__":
    sys.exit(main())
