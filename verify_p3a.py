#!/usr/bin/env python3
"""P3a 验收：确认 XPU 上的 AR 推理后端真的挂上了

分三层，由浅到深：

  第 1 层（秒级）  纯静态，不加载权重
      · resolve_acceleration('xpu', fp32) 是否返回 ('sdpa_varlen_xpu', False)
      · 该模块能否按 create_acceleration 的硬编码拼法导入
      · t2s_engine 的相对路径能否解析
      · t2s_model_cudagraph 惰性层是否生效
      · cuda_graph / flash_attn 是否被正确判为不可用

  第 2 层（--deep，十几秒）  真加载 GPT 权重
      · create_acceleration(...) 返回非 None   ← streaming.py 就是这一句
      · acc.prepare() 成功                     ← 真正 load_decoder + load_state_dict
      这一层过了，gpt_streaming 一定能拿到后端。

  第 3 层（手动）  起 api_v2.py 发请求，命令见脚本末尾。

用法：
    cd D:\\GPSVT\\GPT-SoVITS
    .\\.venv\\Scripts\\python.exe verify_p3a.py
    .\\.venv\\Scripts\\python.exe verify_p3a.py --deep

退出码：0 = 通过；1 = 失败
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS: list[tuple[str, str, str]] = []

CURL_HINT = """
  端到端测试（第 3 层，手动）：

    # 终端 A：起 API
    .\\.venv\\Scripts\\python.exe api_v2.py

    # 终端 B：PowerShell 里跑（用哈希表拼 JSON，省得转义）
    $body = @{
      text              = "这一句会边想边说，我们试试流式。"
      text_lang         = "zh"
      ref_audio_path    = "D:\\ref.wav"
      prompt_text       = "参考音频的文本"
      prompt_lang       = "zh"
      gpt_streaming     = $true
      streaming_mode    = 3
      batch_size        = 1
      media_type        = "wav"
      text_split_method = "cut5"
    } | ConvertTo-Json

    curl.exe -N -X POST http://127.0.0.1:9880/tts `
      -H "Content-Type: application/json" -d $body --output out.wav

  怎么判断流式真生效：
    · 音频分块到达（约 2 秒一块），不是等整句算完再一次性吐
    · 控制台按文本段逐个打印
    · 如果还是整块出来 → 检查模型是不是 v2ProPlus（v2Pro 不支持）
"""


def add(cid: str, name: str, status: str) -> None:
    RESULTS.append((cid, name, status))


def head(t: str) -> None:
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def ok(m: str) -> None:
    print(f"  [OK]   {m}")


def bad(m: str) -> None:
    print(f"  [FAIL] {m}")


def warn(m: str) -> None:
    print(f"  [WARN] {m}")


# ============================================================ 第 1 层

def layer1() -> bool:
    head("[1] 静态检查")

    try:
        import torch
    except Exception as e:
        bad(f"import torch 失败：{type(e).__name__}: {e}")
        add("1-0", "torch 可用", "FAIL")
        return False

    try:
        from tools import device_utils as du
        from tools.acceleration import resolve_acceleration

        dev = du.default_device()
        print(f"         device = {dev}")
    except Exception as e:
        bad(f"import tools 失败：{type(e).__name__}: {e}")
        add("1-0", "tools 可用", "FAIL")
        return False

    # ---- A. resolve_acceleration
    try:
        backend, graph = resolve_acceleration(dev, torch.float32)
        print(f"         resolve_acceleration({dev.type}, fp32) -> ({backend!r}, {graph})")
        if backend == "sdpa_varlen_xpu" and graph is False:
            ok("后端 = sdpa_varlen_xpu，graph = False（预期）")
            add("1-A", "resolve_acceleration", "PASS")
        elif backend is None:
            bad("返回 None —— 补丁没生效，gpt_streaming 会抛 ValueError")
            add("1-A", "resolve_acceleration", "FAIL")
            return False
        else:
            warn(f"后端是 {backend!r}，不是 sdpa_varlen_xpu")
            add("1-A", "resolve_acceleration", "REVIEW")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("1-A", "resolve_acceleration", "FAIL")
        return False

    # ---- B. 按 create_acceleration 的拼法导入
    try:
        mod_path = "GPT_SoVITS.Accel.PyTorch.AR.backends." + backend
        mod = importlib.import_module(mod_path)
        print(f"         import {mod_path}")
        print(f"                -> {mod.__file__}")
        if not hasattr(mod, "T2SDecoder"):
            bad("模块里没有 T2SDecoder —— 后端文件内容不对")
            add("1-B", "后端模块可导入", "FAIL")
            return False
        ok("模块导出 T2SDecoder")
        add("1-B", "后端模块可导入", "PASS")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        print("         常见原因：后端文件放到了 Accel/XPU/ 下，")
        print("         但加载路径硬编码为 GPT_SoVITS.Accel.PyTorch.AR.backends.<name>")
        add("1-B", "后端模块可导入", "FAIL")
        return False

    # ---- C. t2s_engine 的相对路径
    try:
        p = f".backends.{backend.lower().replace('-', '_').replace('cudagraph', 'cuda_graph')}"
        print(f"         engine module_path = {p}")
        if p != ".backends.sdpa_varlen_xpu":
            warn(f"与预期不符：{p}")
            add("1-C", "engine 路径解析", "REVIEW")
        else:
            ok("t2s_engine 的相对路径能对上")
            add("1-C", "engine 路径解析", "PASS")
    except Exception as e:
        bad(str(e))
        add("1-C", "engine 路径解析", "FAIL")

    # ---- D. 惰性兼容层
    try:
        from GPT_SoVITS.AR.models import t2s_model_cudagraph as lazy

        if hasattr(lazy, "__getattr__"):
            ok("兼容层已惰性化（__getattr__ 存在）")
            add("1-D", "惰性兼容层", "PASS")
        else:
            bad("兼容层还是顶层 import，XPU 上会 ImportError")
            add("1-D", "惰性兼容层", "FAIL")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("1-D", "惰性兼容层", "FAIL")

    # ---- E. 能力判定
    try:
        from tools.acceleration import cuda_graph_available, flash_attention_available

        g = cuda_graph_available(dev)
        f = flash_attention_available(dev, torch.float32)
        print(f"         cuda_graph_available = {g}   flash_attention_available = {f}")
        if g is False and f is False:
            ok("图与 flash 都正确判为不可用")
            add("1-E", "能力判定", "PASS")
        else:
            warn("预期都是 False；非 False 说明早退分支没打上")
            add("1-E", "能力判定", "REVIEW")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("1-E", "能力判定", "FAIL")

    return True


# ============================================================ 第 2 层

def layer2() -> bool:
    head("[2] 真加载 GPT 权重（--deep）")
    import torch

    from tools import device_utils as du
    from tools.acceleration import create_acceleration

    candidates = [
        ROOT / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt",
        ROOT / "GPT_SoVITS" / "pretrained_models" /
        "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        ROOT / "GPT_SoVITS" / "pretrained_models" /
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        warn("找不到 GPT 权重，跳过第 2 层")
        warn("先跑：.\\\\.venv\\\\Scripts\\\\python.exe download_models.py")
        add("2-0", "GPT 权重存在", "SKIP")
        return True

    size_gb = weights.stat().st_size / (1024 ** 3)
    print(f"         权重：{weights.relative_to(ROOT)}  ({size_gb:.2f} GB)")
    if size_gb < 0.05:
        bad("权重文件太小（<50MB），可能没下完")
        add("2-0", "GPT 权重完整", "FAIL")
        return False

    dev = du.default_device()

    # ---- A. create_acceleration 非 None（streaming.py 那一句）
    t0 = time.time()
    try:
        acc = create_acceleration(
            weights,
            device=dev,
            dtype=torch.float32,
            max_batch_size=1,
            max_seq_length=2048,
            use_cuda_graph=True,
            use_flash_attention=True,
        )
        ms = (time.time() - t0) * 1000
        if acc is None:
            bad("create_acceleration 返回 None —— gpt_streaming 会抛 ValueError")
            add("2-A", "create_acceleration", "FAIL")
            return False
        ok(f"返回 AccelInference（{ms:.0f} ms）")
        add("2-A", "create_acceleration", "PASS")
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        add("2-A", "create_acceleration", "FAIL")
        return False

    # ---- B. prepare() 真加载
    t0 = time.time()
    try:
        ready = acc.prepare(False, False)      # 关图、关 flash
        dt = time.time() - t0
        print(f"         prepare(False, False) -> {ready}   ({dt:.1f}s)")
        if ready:
            ok("后端就绪，权重加载成功")
            add("2-B", "prepare 加载权重", "PASS")
        else:
            bad("prepare() 返回 False —— 后端没起来")
            add("2-B", "prepare 加载权重", "FAIL")
            return False
    except Exception as e:
        bad(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(limit=4)
        add("2-B", "prepare 加载权重", "FAIL")
        return False
    finally:
        try:
            acc.close()
        except Exception:
            pass

    return True


# ============================================================ 汇总

def summary() -> int:
    head("P3a 验收汇总")
    print(f"  {'ID':<5} {'项目':<26} {'结果'}")
    print("  " + "-" * 46)
    for cid, name, status in sorted(RESULTS):
        print(f"  {cid:<5} {name:<26} {status}")
    print()

    fails = [r for r in RESULTS if r[2] == "FAIL"]
    reviews = [r for r in RESULTS if r[2] == "REVIEW"]
    if fails:
        print(f"  {len(fails)} 项失败：{', '.join(f[0] for f in fails)}")
    else:
        print("  静态项全过。")
    if reviews:
        print(f"  {len(reviews)} 项需留意：{', '.join(r[0] for r in reviews)}")

    print(CURL_HINT)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="真加载 GPT 权重（十几秒）")
    args = ap.parse_args()

    print("P3a 验收（XPU AR 推理后端）")
    print(f"仓库：{ROOT}")

    if not layer1():
        return summary()

    if args.deep:
        layer2()

    return summary()


if __name__ == "__main__":
    sys.exit(main())
