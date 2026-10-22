#!/usr/bin/env python3
"""
XPUGraph 能力探针 —— 决定 P3b（CUDA Graph 的 Intel 等价物）能不能做
======================================================================

CUDA Graph 的最后一道关卡不是"写代码"，而是"这个运行时到底支不支持
你需要的那些语义"。这个脚本在 5 分钟内把 9 个关键能力逐个问一遍，
比盲写 300 行再调试 6 天划算得多。

用法：
    python xpu_graph_probe.py
    python xpu_graph_probe.py --device xpu:0

判读方法见输出末尾的"结论"区块。
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


# ============================================================ 报告基架

PASS, WARN, FAIL = [], [], []
R = "\033[0m"
G = "\033[32m"
Y = "\033[33m"
RD = "\033[31m"
B = "\033[34m"


def ok(name, detail=""):
    PASS.append(name)
    print(f"  {G}[YES ]{R} {name:46s} {detail}")


def warn(name, detail=""):
    WARN.append(name)
    print(f"  {Y}[PART]{R} {name:46s} {detail}")


def bad(name, detail=""):
    FAIL.append(name)
    print(f"  {RD}[NO  ]{R} {name:46s} {detail}")


def section(t):
    print(f"\n{'─' * 76}\n{t}\n{'─' * 76}")


def sync(dev):
    try:
        torch.xpu.synchronize(dev)
    except Exception:
        pass


# ============================================================ 1. 基础探测

def t1_exists(dev):
    section("1. API 存在性")
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        bad("torch.xpu 可用", "没有 XPU，后面全部跳过")
        return False
    ok("torch.xpu 可用")

    found = {}
    for attr in ("XPUGraph", "graph", "graph_pool_handle",
                 "Stream", "Event", "synchronize", "empty_cache"):
        found[attr] = hasattr(torch.xpu, attr)
        (ok if found[attr] else bad)(f"torch.xpu.{attr}")

    if not found["XPUGraph"]:
        bad("XPUGraph 类", "P3b 直接不可做（除非自己封 Level Zero）")
        return False
    return True


# ============================================================ 2. 基本捕获

def t2_basic_capture(dev):
    section("2. 基本捕获 + 重放")
    try:
        g = torch.xpu.XPUGraph()
        static_in = torch.randn(64, 64, device=dev)
        static_out = torch.empty(64, 64, device=dev)

        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream(dev))
        with torch.xpu.stream(s):
            for _ in range(3):                     # warmup
                static_in @ static_in
        torch.xpu.current_stream(dev).wait_stream(s)

        with torch.xpu.graph(g):
            static_out.copy_(static_in @ static_in)
        sync(dev)

        ref = (static_in @ static_in).clone()
        static_out.zero_()
        g.replay()
        sync(dev)
        err = (static_out - ref).abs().max().item()
        if err < 1e-3:
            ok("捕获 + replay 数值正确", f"max_err={err:.2e}")
        else:
            bad("捕获 + replay 数值正确", f"max_err={err:.2e}")
    except Exception as e:
        bad("基本捕获", f"{type(e).__name__}: {str(e)[:80]}")
        return None
    return g


# ============================================================ 3. 输入可变性

def t3_mutable_input(dev):
    section("3. 重放时能否改输入（最关键的可用性前提）")
    try:
        g = torch.xpu.XPUGraph()
        static_in = torch.ones(1024, device=dev)
        static_out = torch.zeros(1024, device=dev)

        with torch.xpu.graph(g):
            static_out.copy_(static_in * 2.0)
        sync(dev)

        r1 = static_out.clone()

        # 改输入缓冲，再 replay —— 这是整个 Accel 设计的核心用法
        static_in.fill_(5.0)
        g.replay()
        sync(dev)
        r2 = static_out.clone()

        if torch.allclose(r1, torch.full_like(r1, 2.0)) and \
           torch.allclose(r2, torch.full_like(r2, 10.0)):
            ok("改缓冲后 replay 生效", "1*2=2, 5*2=10 ✓")
        else:
            bad("改缓冲后 replay 生效",
                f"r1[0]={r1[0].item():.1f} r2[0]={r2[0].item():.1f}（期望 2.0 / 10.0）")
    except Exception as e:
        bad("改缓冲后 replay", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 4. memory pool

def t4_pool(dev):
    section("4. 显存池共享（多张图共用一块显存，避免 VRAM 爆掉）")
    try:
        g1 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g1):
            a = torch.randn(512, 512, device=dev)
            b = a @ a
        sync(dev)

        if hasattr(g1, "pool"):
            try:
                p = g1.pool()
                ok("graph.pool()", f"返回 {type(p).__name__}")
            except Exception as e:
                bad("graph.pool()", f"{type(e).__name__}: {str(e)[:60]}")
                p = None
        else:
            bad("graph.pool()", "无此方法 → 多图会各占一份显存")
            p = None

        # 试着把 pool 传给第二张图
        if p is not None and hasattr(torch.xpu, "graph"):
            try:
                g2 = torch.xpu.XPUGraph()
                with torch.xpu.graph(g2, p):
                    c = torch.randn(256, 256, device=dev)
                    d = c @ c
                sync(dev)
                ok("pool 可传给第二张图", "多 batch size 图可共享显存")
            except TypeError:
                warn("pool 传参签名不同",
                     "torch.xpu.graph 不接受 pool 位置参数，需按版本适配")
            except Exception as e:
                bad("pool 共享", f"{type(e).__name__}: {str(e)[:60]}")
    except Exception as e:
        bad("memory pool 探测", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 5. RNG（致命）

def t5_rng(dev):
    section("5. 图内随机数是否推进（★ 不推进则输出会重复）")
    try:
        g = torch.xpu.XPUGraph()
        static_out = torch.zeros(8, dtype=torch.long, device=dev)
        static_prob = torch.full((1, 1000), 0.001, device=dev)

        with torch.xpu.graph(g):
            static_out.copy_(
                torch.multinomial(static_prob, 8, replacement=True).view(-1)
            )
        sync(dev)

        r1 = static_out.clone()
        g.replay()
        sync(dev)
        r2 = static_out.clone()
        g.replay()
        sync(dev)
        r3 = static_out.clone()

        same12 = torch.equal(r1, r2)
        same23 = torch.equal(r2, r3)

        if same12 and same23:
            bad("图内 RNG 推进",
                "三次采样完全相同 → 必须在图外做采样，或手动管理 RNG 状态")
        elif same12 != same23:
            warn("图内 RNG 推进不稳定", "偶发重复，需在图外做采样")
        else:
            ok("图内 RNG 正常推进", "每次 replay 采样不同")

    except Exception as e:
        bad("图内 RNG 探测", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 6. 多 stream

def t6_stream(dev):
    section("6. 非默认 stream 上捕获")
    try:
        g = torch.xpu.XPUGraph()
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream(dev))

        out = torch.zeros(256, device=dev)
        inp = torch.ones(256, device=dev)

        with torch.xpu.stream(s):
            with torch.xpu.graph(g):
                out.copy_(inp + 1.0)
        torch.xpu.current_stream(dev).wait_stream(s)
        sync(dev)

        g.replay()
        sync(dev)
        ok("自定义 stream 内捕获",
           f"out[0]={out[0].item():.1f}（期望 2.0）" if out[0].item() == 2.0 else
           f"out[0]={out[0].item():.1f}")
    except Exception as e:
        bad("自定义 stream 捕获", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 7. 多尺寸图

def t7_multi_graph(dev):
    section("7. 同一进程内捕获多张不同 shape 的图（对应不同 batch size）")
    sizes = [1, 2, 4, 8]
    graphs, errs = {}, []
    try:
        for n in sizes:
            g = torch.xpu.XPUGraph()
            x = torch.randn(n, 128, device=dev)
            y = torch.empty(n, 128, device=dev)
            with torch.xpu.graph(g):
                y.copy_(x * 3.0)
            sync(dev)
            graphs[n] = (g, x, y)

        # 全部 replay 一遍验证互不干扰
        for n, (g, x, y) in graphs.items():
            x.fill_(float(n))
            g.replay()
            sync(dev)
            exp = n * 3.0
            if abs(y[0, 0].item() - exp) > 1e-3:
                errs.append(f"bs={n} 期望{exp} 实际{y[0,0].item():.1f}")

        if errs:
            bad("多图共存", "; ".join(errs))
        else:
            ok("多图共存", f"{len(sizes)} 张图各自正确")
    except Exception as e:
        bad("多图共存", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 8. 复杂算子

def t8_ops(dev):
    section("8. 图内能否包含训练/推理真实算子")
    ops = {
        "conv1d": lambda: torch.nn.functional.conv1d(
            torch.randn(1, 32, 128, device=dev), torch.randn(32, 32, 3, device=dev), padding=1),
        "layer_norm": lambda: torch.nn.functional.layer_norm(
            torch.randn(1, 64, 128, device=dev), (128,)),
        "index_copy_": lambda: torch.zeros(8, 64, device=dev).index_copy_(
            0, torch.tensor([0, 2]), torch.randn(2, 64, device=dev)),
        "take_along_dim": lambda: torch.take_along_dim(
            torch.randn(2, 10, 8, device=dev), torch.tensor([[[1], [2]]], device=dev), 1),
        "cumsum": lambda: torch.cumsum(torch.randn(4, 100, device=dev), -1),
        "topk": lambda: torch.topk(torch.randn(4, 1000, device=dev), 15).values,
        "conv_transpose1d": lambda: torch.nn.functional.conv_transpose1d(
            torch.randn(1, 32, 32, device=dev), torch.randn(32, 16, 8, device=dev), stride=4, padding=2),
    }
    for name, fn in ops.items():
        try:
            g = torch.xpu.XPUGraph()
            buf = fn()                      # warmup 兼预分配
            with torch.xpu.graph(g):
                out = fn()
            sync(dev)
            g.replay()
            sync(dev)
            ok(f"图内 {name}")
        except Exception as e:
            bad(f"图内 {name}", f"{type(e).__name__}: {str(e)[:60]}")


# ============================================================ 9. 捕获耗时

def t9_timing(dev):
    section("9. capture 耗时（影响启动速度）")
    try:
        t0 = time.perf_counter()
        g = torch.xpu.XPUGraph()
        x = torch.randn(256, 256, device=dev)
        y = torch.empty_like(x)
        with torch.xpu.graph(g):
            for _ in range(20):
                y.copy_(x @ x)
        sync(dev)
        capture_ms = (time.perf_counter() - t0) * 1000

        g.replay()
        sync(dev)
        t1 = time.perf_counter()
        for _ in range(10):
            g.replay()
        sync(dev)
        replay_ms = (time.perf_counter() - t1) / 10 * 1000

        ok("capture 完成", f"{capture_ms:.1f} ms")
        ok("replay 完成", f"{replay_ms:.3f} ms/次")

        if capture_ms > 3000:
            warn("capture 偏慢", "多 batch size 会让启动时间显著变长")
    except Exception as e:
        bad("耗时测量", f"{type(e).__name__}: {str(e)[:80]}")


# ============================================================ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    print(f"{B}XPUGraph 能力探针{R}")
    print(f"torch = {torch.__version__}    device = {dev}")

    try:
        if dev.type == "xpu":
            p = torch.xpu.get_device_properties(dev)
            print(f"GPU   = {getattr(p, 'name', '?')}  "
                  f"{getattr(p, 'total_memory', 0) / 1024**3:.1f}GB  "
                  f"driver={getattr(p, 'driver_version', '?')}")
    except Exception as e:
        print(f"(设备信息读取失败: {e})")

    if not t1_exists(dev):
        print(f"\n{RD}结论：环境不具备 XPUGraph，P3b 不可做。{R}")
        print("建议：只做 P3a（关掉 graph，走 SDPA eager），推理功能完整。")
        return 2

    t2_basic_capture(dev)
    t3_mutable_input(dev)
    t4_pool(dev)
    t5_rng(dev)
    t6_stream(dev)
    t7_multi_graph(dev)
    t8_ops(dev)
    t9_timing(dev)

    # ---------------- 结论
    print(f"\n{'=' * 76}\n结论\n{'=' * 76}")
    print(f"通过 {len(PASS)}  /  部分 {len(WARN)}  /  失败 {len(FAIL)}")

    rng_failed = any("RNG" in f for f in FAIL)
    pool_failed = any("pool" in f for f in FAIL)
    multi_failed = any("多图" in f for f in FAIL)

    print()
    if not FAIL:
        print(f"{G}★ 全部通过：P3b 可以按计划做（预计 2 天移植 + 2 天调试）{R}")
    elif rng_failed:
        print(f"{Y}★ RNG 不推进：这是可解的，但必须改设计{R}")
        print("   做法：把采样（multinomial/topk+multinomial）移出图，")
        print("        图里只跑到 logits，采样在 Python 侧做。")
        print("   代价：每 token 多一次 host-device 同步，收益从 1.8x 降到约 1.3x。")
        print("   影响：P3b 工期 +1 天，但仍然值得。")
    elif pool_failed:
        print(f"{Y}★ 无显存池：能做，但 VRAM 占用会翻几倍{R}")
        print("   做法：限制只捕获 1-2 个 batch size（比如只捕获 bs=1），")
        print("        其余走 eager。收益打折但风险可控。")
    elif multi_failed:
        print(f"{Y}★ 多图共存有问题：只能捕获单张图{R}")
        print("   做法：固定 bs=1 的图，批量模式走 eager。")
    else:
        print(f"{Y}★ 有部分能力缺失，按上面提示降级实现{R}")

    if WARN:
        print(f"\n需要留意的部分支持项：")
        for w in WARN:
            print(f"  · {w}")

    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
