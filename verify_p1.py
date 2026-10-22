#!/usr/bin/env python3
"""P1（基础层）验收脚本 —— Python 版

用法：
    cd D:\\GPSVT\\GPT-SoVITS
    .\\.venv\\Scripts\\python.exe verify_p1.py

检查 P1 规格里的 D1/D2/D3/D5/D6/D7，外加三项额外检查。
D4（推理出声）需要人工，脚本会给提示。

退出码：0 = 全过；1 = 有失败项
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

# ------------------------------------------------------------------ 环境准备

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台是 cp936，直接 print 某些符号会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "spec", "node_modules", "logs", "tmp"}

RESULTS: list[tuple[str, str, str]] = []   # (id, name, status)  status: PASS/FAIL/REVIEW/SKIP


def add(cid: str, name: str, status: str) -> None:
    RESULTS.append((cid, name, status))


def head(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def iter_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def scan(pattern: str, exclude: tuple[str, ...] = ()) -> list[tuple[str, int, str]]:
    """返回 (相对路径, 行号, 行内容) 列表。"""
    rx = re.compile(pattern)
    hits = []
    for p in iter_py_files():
        rel = str(p.relative_to(ROOT))
        if any(e in rel for e in exclude):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append((rel, i, line.strip()))
    return hits


# ------------------------------------------------------------------ D1
def check_d1() -> None:
    head("[D1] 无 CUDA 硬编码残留")
    hits = scan(r"torch\.cuda\.", exclude=("tools/device_utils.py", "tools\\device_utils.py"))
    if not hits:
        ok("0 处残留")
        add("D1", "无 CUDA 残留", "PASS")
        return

    for rel, ln, line in hits[:20]:
        print(f"         {rel}:{ln}  {line[:100]}")
    if len(hits) > 20:
        print(f"         ... 共 {len(hits)} 处")

    # if 守卫里的判断不算问题
    guarded = [h for h in hits if re.search(r"\bif\b.*\bcuda\b", h[2], re.I)]
    rest = [h for h in hits if h not in guarded]
    if not rest:
        ok(f"{len(hits)} 处都在 if 守卫里，可接受")
        add("D1", "无 CUDA 残留", "PASS")
    else:
        warn(f"{len(rest)} 处非守卫调用，需人工确认")
        add("D1", "无 CUDA 残留", "REVIEW")


# ------------------------------------------------------------------ D2
def check_d2() -> None:
    head("[D2] 设备识别")
    try:
        from tools import device_utils as du

        desc = du.describe()
        print(f"         {desc}")
        has_a750 = "A750" in desc
        has_dgpu = "dGPU" in desc
        has_bf16_false = "bf16=False" in desc
        if has_a750 and has_dgpu and has_bf16_false:
            ok("A750 / dGPU / bf16=False 全部匹配")
            add("D2", "设备识别", "PASS")
        else:
            miss = [
                n for n, v in (
                    ("A750", has_a750),
                    ("dGPU", has_dgpu),
                    ("bf16=False", has_bf16_false),
                ) if not v
            ]
            bad(f"未匹配：{miss}")
            add("D2", "设备识别", "FAIL")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("D2", "设备识别", "FAIL")


# ------------------------------------------------------------------ D3
def check_d3() -> None:
    head("[D3] 基础模块自检（tools/_selftest.py）")
    st = ROOT / "tools" / "_selftest.py"
    if not st.exists():
        bad("找不到 tools/_selftest.py")
        add("D3", "自检", "FAIL")
        return
    try:
        r = subprocess.run(
            [sys.executable, str(st)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), timeout=300,
        )
        tail = (r.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            print(f"         {line}")
        if r.returncode == 0:
            ok("exit=0")
            add("D3", "自检 60 项", "PASS")
        else:
            bad(f"exit={r.returncode}")
            if r.stderr:
                print("         " + (r.stderr.strip().splitlines()[-1][:150]))
            add("D3", "自检 60 项", "FAIL")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("D3", "自检", "FAIL")


# ------------------------------------------------------------------ D7
def check_d7() -> None:
    head("[D7] config.py 选对后端与精度")
    try:
        import config

        dev = config.infer_device
        is_half = config.is_half
        print(f"         infer_device = {dev}  (type={dev.type})")
        print(f"         is_half      = {is_half}")
        print(f"         GPU_INFOS    = {config.GPU_INFOS}")

        dev_ok = dev.type == "xpu"
        half_ok = is_half is False
        if dev_ok and half_ok:
            ok("device=xpu, is_half=False")
            add("D7", "config 后端/精度", "PASS")
        else:
            if not dev_ok:
                bad(f"device 应为 xpu，实际 {dev.type}")
            if not half_ok:
                bad("is_half 应为 False（本机 fp32）—— 检查 get_device_dtype_sm 是否真的替换了")
            add("D7", "config 后端/精度", "FAIL")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        add("D7", "config 后端/精度", "FAIL")


# ------------------------------------------------------------------ D5
def check_d5() -> None:
    head("[D5] dtype 解析落到 fp32")
    try:
        from tools import device_utils as du
        from tools import dtype_utils as dt

        d = du.default_device()
        resolved = dt.resolve(True, d)          # 故意请求低精度
        fp16_fast = du.supports_fp16_fast(d)
        print(f"         device          = {d}")
        print(f"         resolve(True)    = {dt.name(resolved)}")
        print(f"         supports_fp16_fast = {fp16_fast}")

        if resolved == __import__("torch").float32:
            ok("即使请求低精度也降级到 fp32")
            add("D5", "dtype 落 fp32", "PASS")
        else:
            bad(f"应为 fp32，实际 {dt.name(resolved)}")
            add("D5", "dtype 落 fp32", "FAIL")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("D5", "dtype 落 fp32", "FAIL")


# ------------------------------------------------------------------ D6
SHIM_FILES = [
    "GPT_SoVITS/s2_train.py",
    "GPT_SoVITS/s2_train_v3.py",
    "GPT_SoVITS/s2_train_v3_lora.py",
    "GPT_SoVITS/s1_train.py",
    "GPT_SoVITS/inference_webui_fast.py",
    "GPT_SoVITS/prepare_datasets/1-get-text.py",
    "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
    "GPT_SoVITS/prepare_datasets/2-get-sv.py",
    "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
]


def check_d6() -> None:
    head("[D6] 路径垫片完整性（9 个入口脚本）")
    missing = []
    for rel in SHIM_FILES:
        p = ROOT / rel
        if not p.exists():
            bad(f"文件不存在  {rel}")
            missing.append(rel)
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "sys.path.insert" in text:
            ok(rel)
        else:
            bad(f"无垫片      {rel}")
            missing.append(rel)

    # prepare_datasets 必须是 parents[2]
    for rel in SHIM_FILES[5:]:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "parents[2]" not in text and "parents[1]" in text:
            warn(f"{rel} 用了 parents[1]，prepare_datasets 应该是 parents[2]")

    add("D6", "9 个垫片", "PASS" if not missing else "FAIL")


# ------------------------------------------------------------------ X1
def check_x1() -> None:
    head("[X1] _CUDA_VISIBLE_DEVICES 硬编码残留")
    hits = scan(r'["\']_CUDA_VISIBLE_DEVICES["\']\s*\]\s*=', exclude=("device_utils.py",))
    if not hits:
        ok("已无硬编码赋值，应已改为 du.set_visible_devices")
        return
    for rel, ln, line in hits[:10]:
        print(f"         {rel}:{ln}  {line[:100]}")
    warn(f"共 {len(hits)} 处，确认是否已改")


# ------------------------------------------------------------------ X2
def check_x2() -> None:
    head("[X2] GPT_SoVITS/configs/tts_infer.yaml")
    p = ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if not p.exists():
        warn("文件不存在")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^custom:\s*$([\s\S]*?)(?=^\w|$)", text, re.M)
    block = m.group(1) if m else text
    dev = re.search(r"device:\s*(\S+)", block)
    flag = re.search(r"is_half:\s*(\S+)", block)
    print(f"         custom.device  = {dev.group(1) if dev else '(未找到)'}")
    print(f"         custom.is_half = {flag.group(1) if flag else '(未找到)'}")
    if dev and dev.group(1).strip() == "xpu":
        ok("device: xpu")
    else:
        warn("custom.device 不是 xpu")


# ------------------------------------------------------------------ X3
def check_x3() -> None:
    head("[X3] git 改动清单")
    try:
        r = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), timeout=30,
        )
        lines = (r.stdout or "").strip().splitlines()
        if not lines:
            print("         （工作区干净，或不是 git 仓库）")
        for line in lines[:40]:
            print(f"         {line}")
        if len(lines) > 40:
            print(f"         ... 共 {len(lines)} 项")
    except Exception as e:
        warn(f"git 不可用：{e}")


# ------------------------------------------------------------------ 汇总
def summary() -> int:
    head("P1 验收汇总")
    print(f"  {'ID':<4} {'项目':<22} {'结果'}")
    print("  " + "-" * 44)
    for cid, name, status in sorted(RESULTS):
        print(f"  {cid:<4} {name:<22} {status}")
    print()

    fails = [r for r in RESULTS if r[2] == "FAIL"]
    reviews = [r for r in RESULTS if r[2] == "REVIEW"]

    if fails:
        print(f"  有 {len(fails)} 项失败：{', '.join(f[0] for f in fails)}")
    else:
        print("  脚本项全过。")
    if reviews:
        print(f"  {len(reviews)} 项需人工确认：{', '.join(r[0] for r in reviews)}")

    print()
    print("  还剩 D4 需要人工：")
    print()
    print("      .\\.venv\\Scripts\\python.exe GPT_SoVITS\\inference_webui.py zh_CN")
    print()
    print("      浏览器开 9872，上传参考音频，合成一句中文。")
    print("      通过标准：有音频输出、控制台无 Traceback。")
    print()

    return 1 if fails else 0


def main() -> int:
    print("P1 验收（基础层：设备抽象 + fp32）")
    print(f"仓库：{ROOT}")
    print(f"Python：{sys.version.split()[0]}  ({sys.executable})")

    # torch 先探一下，避免后面全是连带错误
    try:
        import torch
        print(f"torch：{torch.__version__}  xpu_available={torch.xpu.is_available()}")
    except Exception as e:
        print(f"[FATAL] 导入 torch 失败：{e}")
        print("        先在 venv 里装好 torch XPU 版")
        return 1

    for fn in (check_d1, check_d2, check_d3, check_d7, check_d5, check_d6, check_x1, check_x2, check_x3):
        try:
            fn()
        except Exception:
            print()
            print(f"[ERROR] {fn.__name__} 崩了：")
            traceback.print_exc(limit=3)
            add(fn.__name__, "内部错误", "FAIL")

    return summary()


if __name__ == "__main__":
    sys.exit(main())
