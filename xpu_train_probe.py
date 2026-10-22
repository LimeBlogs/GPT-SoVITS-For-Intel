#!/usr/bin/env python3
"""
GPT-SoVITS Intel XPU 训练可行性探针
===================================

在 Intel 显卡机器上跑这个脚本，30 秒内就能知道：
    - 哪些训练必需算子能跑
    - 哪些能跑但慢得离谱
    - 哪些直接不可用（需要改方案）

用法：
    python xpu_train_probe.py                # 默认测 xpu，与 CPU 做对照
    python xpu_train_probe.py --device cpu   # 只测 CPU
    python xpu_train_probe.py --dtype bf16   # 只测 bf16
    python xpu_train_probe.py --device xpu:0 # 指定某张卡

依赖：只要 torch（不需要装 gsv-src）
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ 工具

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

RESULTS: list[dict] = []


def record(case, dtype, status, ms=None, note=""):
    RESULTS.append({"case": case, "dtype": str(dtype).replace("torch.", ""),
                    "status": status, "ms": ms, "note": note})
    color = {"OK": GREEN, "SLOW": YELLOW, "FAIL": RED, "SKIP": BLUE}[status]
    ms_s = f"{ms:8.2f} ms" if ms is not None else "        "
    print(f"  {color}[{status:4s}]{RESET} {case:38s} {ms_s}  {note}")


def run_case(name, fn, dtype, device, there_and_back=True, threshold_ms=None):
    """执行一个测试用例。

    there_and_back=True 会额外测 backward，因为训练光 forward 能跑没用。
    """
    t = name + (" + 反传" if there_and_back else "")
    try:
        if there_and_back:
            ms = _timed(fn, dtype=dtype, device=device, backward=True)
        else:
            ms = _timed(fn, dtype=dtype, device=device, backward=False)
    except Exception as e:
        record(t, dtype, "FAIL", None, f"{type(e).__name__}: {str(e)[:70]}")
        return False

    if threshold_ms is not None and ms > threshold_ms:
        record(t, dtype, "SLOW", ms, f"超过 {threshold_ms:.0f}ms 阈值")
        return True
    record(t, dtype, "OK", ms)
    return True


def _timed(fn, dtype, device, backward, warmup=2, iters=3):
    for _ in range(warmup):
        out = fn(dtype=dtype, device=device)
        if backward and isinstance(out, torch.Tensor) and out.requires_grad:
            out.backward()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(dtype=dtype, device=device)
        if backward and isinstance(out, torch.Tensor) and out.requires_grad:
            out.backward()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def _sync(device):
    d = torch.device(device)
    try:
        if d.type == "cuda":
            torch.cuda.synchronize()
        elif d.type == "xpu":
            torch.xpu.synchronize()
        elif d.type == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


# ============================================================ 各用例
# 形状参照 GPT-SoVITS 真实训练参数：
#   sampling_rate=32000 / filter_length(n_fft)=1280 / hop=320 / win=1280
#   n_mel=128 / segment_size=32*2048

SR = 32000
N_FFT = 1280
HOP = 320
WIN = 1280
N_MEL = 128
BATCH = 4
SEG_LEN = 32 * 2048 // HOP     # ≈ 205 帧


def case_matmul(dtype, device):
    a = torch.randn(2048, 512, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(512, 512, device=device, dtype=dtype, requires_grad=True)
    return (a @ b).sum()


def case_conv1d(dtype, device):
    x = torch.randn(BATCH, 192, 512, device=device, dtype=dtype, requires_grad=True)
    c = nn.Conv1d(192, 192, 3, padding=1).to(device, dtype)
    return c(x).sum()


def case_conv_transpose1d(dtype, device):
    """SoVITS Generator 的上采样核心。"""
    x = torch.randn(BATCH, 512, 64, device=device, dtype=dtype, requires_grad=True)
    c = nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4).to(device, dtype)
    return c(x).sum()


def case_layer_norm(dtype, device):
    x = torch.randn(BATCH, 205, 192, device=device, dtype=dtype, requires_grad=True)
    return F.layer_norm(x, (192,)).sum()


def case_group_norm(dtype, device):
    x = torch.randn(BATCH, 512, 128, device=device, dtype=dtype, requires_grad=True)
    return F.group_norm(x, 32).sum()


def case_instance_norm(dtype, device):
    x = torch.randn(BATCH, 128, 205, device=device, dtype=dtype, requires_grad=True)
    return F.instance_norm(x).sum()


def case_gru(dtype, device):
    """AR 模型里有 nn.GRU，XPU 上 RNN 支持历来是最弱的一环。"""
    x = torch.randn(205, BATCH, 192, device=device, dtype=dtype, requires_grad=True)
    g = nn.GRU(192, 192, num_layers=1, batch_first=False).to(device, dtype)
    y, _ = g(x)
    return y.sum()


def case_interpolate(dtype, device):
    x = torch.randn(BATCH, 128, 100, device=device, dtype=dtype, requires_grad=True)
    return F.interpolate(x, scale_factor=2, mode="nearest").sum()


def case_avg_pool1d(dtype, device):
    x = torch.randn(BATCH, 64, 512, device=device, dtype=dtype, requires_grad=True)
    return F.avg_pool1d(x, 4, stride=2).sum()


def case_topk(dtype, device):
    x = torch.randn(BATCH, 1024, device=device, dtype=dtype, requires_grad=True)
    return torch.topk(x, 15).values.sum()


def case_softmax_scan(dtype, device):
    """AR 采样里的累积分布采样（真实实现用的是 cumsum + searchsorted）。"""
    logits = torch.randn(BATCH, 1024, device=device, dtype=dtype, requires_grad=True)
    p = F.softmax(logits, dim=-1)
    cdf = torch.cumsum(p, dim=-1)
    return cdf.sum()


def case_multinomial(dtype, device):
    logits = torch.randn(BATCH, 1024, device=device, dtype=torch.float32, requires_grad=False)
    return torch.multinomial(F.softmax(logits, -1), 1).float().sum()


def case_stft_forward_only(dtype, device):
    """按 mel_processing.py 的真实签名复现。"""
    y = torch.randn(BATCH, SR * 2, device=device, dtype=dtype, requires_grad=True)
    pad = int((N_FFT - HOP) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    win = torch.hann_window(WIN, device=device, dtype=dtype)
    spec = torch.stft(
        y, N_FFT, hop_length=HOP, win_length=WIN, window=win,
        center=False, pad_mode="reflect", normalized=False,
        onesided=True, return_complex=False,     # ← 关键：非 complex
    )
    return spec.sum()


def case_stft_backward(dtype, device):
    """训练必需：对 stft 输出求导（梯度要能传回 y_hat）。"""
    y = torch.randn(BATCH, SR * 2, device=device, dtype=dtype, requires_grad=True)
    pad = int((N_FFT - HOP) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    win = torch.hann_window(WIN, device=device, dtype=dtype)
    spec = torch.stft(
        y, N_FFT, hop_length=HOP, win_length=WIN, window=win,
        center=False, pad_mode="reflect", normalized=False,
        onesided=True, return_complex=False,
    )
    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    return mag.sum()


def case_full_mel_loss(dtype, device):
    """复现 s2_train.py:414 那条 mel loss 的完整链路。"""
    y_hat = torch.randn(BATCH, SR * 2, device=device, dtype=dtype, requires_grad=True)
    pad = int((N_FFT - HOP) / 2)
    y2 = F.pad(y_hat.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    win = torch.hann_window(WIN, device=device, dtype=dtype)
    spec = torch.stft(
        y2, N_FFT, hop_length=HOP, win_length=WIN, window=win,
        center=False, pad_mode="reflect", normalized=False,
        onesided=True, return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    mel_basis = torch.randn(N_MEL, N_FFT // 2 + 1, device=device, dtype=dtype)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.mean()


def case_gan_mini_step(dtype, device):
    """最简化的一次 GAN 训练 step：生成器 + 判别器 + L1 + 反传。"""
    g = nn.Sequential(
        nn.Conv1d(192, 256, 7, padding=3),
        nn.LeakyReLU(0.1),
        nn.ConvTranspose1d(256, 1, 16, stride=8, padding=4),
    ).to(device, dtype)
    d = nn.Sequential(
        nn.Conv1d(1, 32, 15, padding=7),
        nn.LeakyReLU(0.1),
        nn.Conv1d(32, 1, 1),
    ).to(device, dtype)
    z = torch.randn(BATCH, 192, 64, device=device, dtype=dtype)
    y = g(z)
    fake = d(y)
    real = d(torch.randn_like(y))
    loss = F.mse_loss(fake, torch.zeros_like(fake)) + F.mse_loss(real, torch.ones_like(real))
    loss.backward()
    return loss.detach()


def case_autocast(dtype, device):
    """AMP 上下文：bf16 训练实际走的是这条。"""
    dev_type = torch.device(device).type
    x = torch.randn(BATCH, 192, 256, device=device, dtype=torch.float32, requires_grad=True)
    c = nn.Conv1d(192, 192, 3, padding=1).to(device)
    with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16):
        y = c(x)
    return y.float().sum()


def case_dataloader_pin(dtype, device):
    """DataLoader + pin_memory 是否支持 xpu。"""
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(64, 8))
    try:
        dl = DataLoader(ds, batch_size=8, num_workers=0,
                        pin_memory=True, pin_memory_device=str(device).split(":")[0])
    except TypeError:
        dl = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=True)
    for xb, in dl:
        if torch.device(device).type != "cpu":
            xb = xb.to(device, non_blocking=True)
    return torch.zeros((), device=device)


def case_ddp_probe(dtype, device):
    """oneCCL 是否装了（多卡才需要，单卡可跳过）。"""
    try:
        import torch.distributed as dist
        if not dist.is_available():
            raise RuntimeError("torch.distributed 不可用")
        backend = "ccl" if torch.device(device).type == "xpu" else "nccl"
        if not dist.is_backend_available(backend):
            raise RuntimeError(f"backend '{backend}' 不可用（单卡训练可忽略）")
    except Exception as e:
        raise e
    return torch.zeros((), device=device)


# ============================================================ 主流程

CASES = [
    ("矩阵乘 (Linear 核心)",            case_matmul,              False, None),
    ("Conv1d",                          case_conv1d,              True,  None),
    ("ConvTranspose1d (生成器上采样)",  case_conv_transpose1d,    True,  None),
    ("LayerNorm",                       case_layer_norm,          True,  None),
    ("GroupNorm",                       case_group_norm,          True,  None),
    ("InstanceNorm",                    case_instance_norm,       True,  None),
    ("GRU (AR 模型依赖)",               case_gru,                 True,  None),
    ("interpolate",                     case_interpolate,         True,  None),
    ("avg_pool1d",                      case_avg_pool1d,          True,  None),
    ("topk",                            case_topk,                False, None),
    ("softmax+cumsum (AR 采样)",        case_softmax_scan,        False, None),
    ("multinomial",                     case_multinomial,         False, None),
    ("STFT forward",                    case_stft_forward_only,   False, None),
    ("STFT 反传 ★训练必需",             case_stft_backward,       True,  None),
    ("完整 mel loss 链路 ★",            case_full_mel_loss,       True,  None),
    ("GAN mini step (G+D) ★",           case_gan_mini_step,       True,  None),
    ("autocast bf16",                   case_autocast,            True,  None),
    ("DataLoader pin_memory",           case_dataloader_pin,      False, None),
    ("oneCCL (多卡，单卡可忽略)",       case_ddp_probe,           False, None),
]


def device_info(device):
    d = torch.device(device)
    print(f"\n{'=' * 78}")
    print(f"设备: {device}")
    print(f"torch: {torch.__version__}")
    try:
        if d.type == "cuda":
            p = torch.cuda.get_device_properties(d)
            print(f"  name={p.name}  mem={p.total_memory / 1024**3:.1f}GB  "
                  f"sm={torch.cuda.get_device_capability(d)}")
            print(f"  cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
        elif d.type == "xpu":
            p = torch.xpu.get_device_properties(d)
            for attr in ("name", "total_memory", "max_compute_units",
                         "driver_version", "architecture", "has_fp16",
                         "has_bfloat16_conversions",
                         "has_subgroup_matrix_multiply_accumulate"):
                if hasattr(p, attr):
                    v = getattr(p, attr)
                    if attr == "total_memory":
                        v = f"{v / 1024**3:.1f}GB"
                    print(f"  {attr}={v}")
    except Exception as e:
        print(f"  (设备信息读取失败: {e})")
    print(f"{'=' * 78}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="xpu / xpu:0 / cuda:0 / cpu")
    ap.add_argument("--dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="只测指定精度；默认 fp32 + bf16")
    ap.add_argument("--no-compare", action="store_true", help="不与 CPU 做耗时对照")
    args = ap.parse_args()

    # ---- 选设备
    if args.device:
        device = args.device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if torch.device(device).type != "cpu" and not (
        (torch.device(device).type == "xpu" and torch.xpu.is_available())
        or (torch.device(device).type == "cuda" and torch.cuda.is_available())
    ):
        print(f"{RED}设备 {device} 不可用，退出。{RESET}")
        print("提示：Intel 卡需要 pip install torch --index-url "
              "https://download.pytorch.org/whl/xpu")
        return 2

    device_info(device)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtypes = [dtype_map[args.dtype]] if args.dtype else [torch.float32]
    # fp32 之外再加一个低精度（Intel 优先 bf16）
    if not args.dtype:
        dtypes.append(torch.bfloat16)

    for dt in dtypes:
        print(f"---- dtype = {str(dt).replace('torch.', '')} ----")
        for name, fn, needs_grad, thresh in CASES:
            run_case(name, fn, dt, device,
                     there_and_back=needs_grad, threshold_ms=thresh)
        print()

    # ---- CPU 对照（只挑关键几项）
    if not args.no_compare and torch.device(device).type != "cpu":
        print(f"---- CPU 对照（同 dtype=fp32）----")
        for name, fn, needs_grad, _ in CASES:
            if "★" not in name and "oneCCL" not in name:
                continue
            try:
                ms = _timed(fn, torch.float32, "cpu", backward=needs_grad,
                            warmup=1, iters=1)
                record(name + " @CPU", torch.float32, "OK", ms)
            except Exception as e:
                record(name + " @CPU", torch.float32, "FAIL", None,
                       f"{type(e).__name__}: {str(e)[:60]}")
        print()

    # ---- 汇总
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    slows = [r for r in RESULTS if r["status"] == "SLOW"]

    print(f"{'=' * 78}")
    print("结论")
    print(f"{'=' * 78}")

    fatal = [r for r in fails if "★" in r["case"] or "GRU" in r["case"]]
    if fatal:
        print(f"{RED}✗ 有阻断项，P4 无法直接做：{RESET}")
        for r in fatal:
            print(f"    {r['case']} [{r['dtype']}]  {r['note']}")
        print()
        print("  应对：")
        print("    · STFT 反传不可用 → 把 mel loss 段包进 torch.fft fallback，")
        print("      或对 y_hat 先 .cpu() 算 mel 再送回（损失 30-50% 速度）")
        print("    · GRU 不可用 → 用 nn.Linear 手工实现单步 GRU，或整段搬 CPU")
        print()
    elif fails:
        print(f"{YELLOW}△ 有非阻断失败项（多卡/边缘算子）：{RESET}")
        for r in fails:
            print(f"    {r['case']} [{r['dtype']}]  {r['note']}")
        print()
    else:
        print(f"{GREEN}✓ 全部通过，P4 可以按计划做（机械替换为主）{RESET}\n")

    if slows:
        print(f"{YELLOW}慢得可疑（建议用 CPU 兜底这几项）：{RESET}")
        for r in slows:
            print(f"    {r['case']} [{r['dtype']}]  {r['ms']:.1f} ms")
        print()

    # 估算训练时间（粗算：STFT+GAN step 的耗时 × 预期步数）
    xpu_steps = [r for r in RESULTS if r["case"].startswith("GAN mini step")
                 and r["status"] == "OK" and "CPU" not in r["case"]]
    if xpu_steps and torch.device(device).type != "cpu":
        ms = xpu_steps[-1]["ms"]
        # 真实训练 batch 更大、层更多，粗估 8-15x
        est_s = ms * 10 / 1000
        print(f"粗估：真实训练每 step ≈ {est_s:.2f}s，"
              f"5000 step ≈ {est_s * 5000 / 3600:.1f} 小时")
        print("      （仅为量级参考，实际受 IO / 数据加载影响会更大）\n")

    print(f"明细已记录：{len(RESULTS)} 项（FAIL {len(fails)} / SLOW {len(slows)}）")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
