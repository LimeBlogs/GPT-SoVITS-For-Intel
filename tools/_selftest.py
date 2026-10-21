#!/usr/bin/env python3
"""tools 下三个基础模块的自检脚本。

不依赖 gsv-src 的其余部分，也不需要 GPU。

用法：
    cd gsv-src && python tools/_selftest.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from tools import device_utils as du
from tools import amp_utils as amp
from tools import dtype_utils as dt


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n--- {title} ---")


# ============================================================ device_utils

section("device_utils 基础")
print(f"  backend_type() = {du.backend_type()}")
print(f"  default_device() = {du.default_device()}")
print(f"  describe() = {du.describe()}")

check("available_accelerators 返回 list", isinstance(du.available_accelerators(), list))
check("backend_type 是合法值",
      du.backend_type() in ("cuda", "xpu", "mps", "cpu"))
check("accel_available('cpu') 恒 True", du.accel_available("cpu") is True)
check("default_device 是 torch.device",
      isinstance(du.default_device(), torch.device))

p = du.properties(0)
check("properties 含必需字段",
      all(k in p for k in ("name", "total_memory_gb", "compute_units",
                           "has_fp16", "has_bf16", "has_xmx", "is_integrated")),
      f"name={p['name']}")

section("device_utils 运行时操作（不应抛异常）")
cpu = torch.device("cpu")
for fn in (lambda: du.empty_cache(cpu),
           lambda: du.synchronize(cpu),
           lambda: du.manual_seed_all(42),
           lambda: du.memory_allocated_gb(cpu),
           lambda: du.peak_memory_gb(cpu),
           lambda: du.reset_peak_memory(cpu),
           lambda: du.memory_report(cpu, "test"),
           lambda: du.apply_env_defaults()):
    try:
        fn()
        ok = True
        err = ""
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
    check("无异常", ok, err)

check("current_stream(cpu) 返回 None", du.current_stream(cpu) is None)
check("new_event(cpu) 返回 None", du.new_event(cpu) is None)

section("device_utils 可见性")
name = du.visible_env_name()
check("visible_env_name 合法", name in ("_CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"), name)
import os
du.set_visible_devices("0")
check("set_visible_devices 写入了环境变量", os.environ.get(name) == "0")
os.environ.pop(name, None)
os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)

# ============================================================ dtype_utils

section("dtype_utils.parse")
cases = [
    (True, torch.float16),
    (False, torch.float32),
    ("bf16", torch.bfloat16),
    ("Bfloat16", torch.bfloat16),
    ("fp16", torch.float16),
    ("half", torch.float16),
    ("fp32", torch.float32),
    ("False", torch.float32),
    ("true", torch.float16),
    (None, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    ("garbage", torch.float32),
]
for inp, want in cases:
    got = dt.parse(inp)
    check(f"parse({inp!r}) == {want}", got == want, f"got {got}")

section("dtype_utils 其它")
check("is_half(bf16) 为 True", dt.is_half(torch.bfloat16) is True)
check("is_half(fp32) 为 False", dt.is_half(torch.float32) is False)
check("name(bf16) == 'bf16'", dt.name(torch.bfloat16) == "bf16")

m = torch.nn.Linear(4, 4)
check("apply(fp32) 不改 dtype",
      dt.apply(m, torch.float32).weight.dtype == torch.float32)
check("apply(bf16) 改了 dtype",
      dt.apply(torch.nn.Linear(4, 4), torch.bfloat16).weight.dtype == torch.bfloat16)

t = torch.randn(4)
check("cast 支持 fp32→bf16", dt.cast(t, torch.bfloat16).dtype == torch.bfloat16)
i = torch.tensor([1, 2, 3])
check("cast 不动整型", dt.cast(i, torch.bfloat16).dtype == torch.int64)

# resolve 在 CPU 上必须返回 fp32
check("resolve(bf16, cpu) == fp32",
      dt.resolve(torch.bfloat16, "cpu") == torch.float32)
check("resolve(True, cpu) == fp32", dt.resolve(True, "cpu") == torch.float32)

# ============================================================ amp_utils

section("amp_utils.autocast_ctx")
import contextlib

for dev, en, fp32 in [("cpu", True, False), ("cpu", False, False),
                      ("cpu", True, True)]:
    try:
        with amp.autocast_ctx(dev, en, force_fp32=fp32):
            x = torch.randn(2, 2) @ torch.randn(2, 2)
        ok, err = True, ""
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    check(f"ctx({dev}, enabled={en}, force_fp32={fp32}) 可用", ok, err)

section("amp_utils.make_scaler")
for dev in ("cpu",):
    # 注意：loss 必须挂在 requires_grad 的参数上，否则 backward() 会报
    # "element 0 of tensors does not require grad"。
    # 用 ones 而不是 zeros：zeros 处梯度为 0，参数不会变化，测不出 step 是否生效。
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.SGD([p], lr=0.1)
    try:
        s = amp.make_scaler(dev, True)
        loss = (p ** 2).sum()
        s.scale(loss).backward()
        s.unscale_(opt)
        before = p.detach().clone()
        s.step(opt)
        s.update()
        ok = not torch.equal(before, p.detach())
        err = "" if ok else "参数未被 step 更新"
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    check(f"scaler 生命周期可用 ({dev})", ok, err)

section("amp_utils 精度工具")
mod = torch.nn.Module()
mod.flow = torch.nn.Linear(2, 2).to(torch.bfloat16)
mod.enc_q = torch.nn.Linear(2, 2).to(torch.bfloat16)
mod.main = torch.nn.Linear(2, 2).to(torch.bfloat16)
amp.selective_fp32(mod)
check("selective_fp32 把 flow 拉回 fp32", mod.flow.weight.dtype == torch.float32)
check("selective_fp32 把 enc_q 拉回 fp32", mod.enc_q.weight.dtype == torch.float32)
check("selective_fp32 不动 main", mod.main.weight.dtype == torch.bfloat16)

b = (torch.randn(2, 3), torch.tensor([1, 2]), ["x"])
out = amp.cast_batch(b, torch.bfloat16, torch.device("cpu"))
check("cast_batch 浮点转 bf16", out[0].dtype == torch.bfloat16)
check("cast_batch 整型保持", out[1].dtype == torch.int64)
check("cast_batch 非张量原样", out[2] == ["x"])

check("all_finite 正常张量", amp.all_finite(torch.randn(3)) is True)
nan = torch.tensor([1.0, float("nan")])
check("all_finite 检出 NaN", amp.all_finite(nan) is False)

# ============================================================ 汇总

total = len(PASS) + len(FAIL)
print(f"\n{'=' * 60}")
if FAIL:
    print(f"\033[31m{len(FAIL)} / {total} 失败\033[0m")
    for f in FAIL:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"\033[32m全部通过：{total} 项\033[0m")

if not du.accel_available("xpu"):
    print("\n\033[33m提示：当前环境没有 XPU，仅验证了 CPU 兜底路径。\033[0m")
    print("     到 Intel 机器上再跑一次，才会覆盖 xpu 分支。")
