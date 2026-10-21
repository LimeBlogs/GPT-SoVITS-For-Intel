"""Device-agnostic helpers. Single source of truth for accelerator access.

Supported: cuda / xpu (Intel) / mps (Apple) / cpu

设计原则
--------
1. 全项目不许再直接写 torch.cuda.*，一律走这里。
2. 每个函数都必须有 CPU 兜底，任何加速器缺失都不许抛异常。
3. 能力探测一律"运行时实测 + getattr 兜底"，不靠版本号猜。

Intel 独显补充说明
------------------
Arc A/B 系独显有独立显存（VRAM），不是 UMA 共享内存，所以：
  - total_memory 就是真实显存，可以放心用作门槛判断
  - 不需要为宿主内存压力预留余量（核显才需要）
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import torch


# ---------------------------------------------------------------- 后端探测

def available_accelerators() -> list[str]:
    """返回当前可用的加速后端列表，按优先级排序。"""
    found = []
    if torch.cuda.is_available() and torch.version.cuda is not None:
        found.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        found.append("mps")
    return found


def accel_available(dev) -> bool:
    """指定 device 类型当前是否真的可用。

    注意 cuda 分支额外检查 torch.version.cuda is not None：
    ROCm 版 torch 的 torch.cuda.is_available() 也是 True，但 version.cuda 为 None。
    """
    t = torch.device(dev).type
    if t == "cuda":
        return torch.cuda.is_available() and torch.version.cuda is not None
    if t == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if t == "mps":
        return bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    return t == "cpu"


def backend_type() -> str:
    """当前实际使用的加速后端类型。训练脚本全局只用这一个。"""
    for name in ("cuda", "xpu", "mps"):
        if accel_available(name):
            return name
    return "cpu"


def default_device(prefer: Optional[str] = None) -> torch.device:
    """优先顺序：显式 prefer > GSV_DEVICE 环境变量 > XPU > CUDA > MPS > CPU。"""
    prefer = prefer or os.environ.get("GSV_DEVICE")
    if prefer:
        return torch.device(prefer)
    for name in ("xpu", "cuda", "mps"):
        if accel_available(name):
            return torch.device(name)
    return torch.device("cpu")


def device_count(dev_type: Optional[str] = None) -> int:
    dev_type = dev_type or backend_type()
    if dev_type == "cuda":
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    if dev_type == "xpu":
        return torch.xpu.device_count() if accel_available("xpu") else 0
    return 0


def set_device(idx: int, dev_type: Optional[str] = None) -> None:
    dev_type = dev_type or backend_type()
    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(idx)
    elif dev_type == "xpu" and accel_available("xpu"):
        torch.xpu.set_device(idx)


# ---------------------------------------------------------------- 硬件能力

def properties(idx: int = 0, dev_type: Optional[str] = None) -> dict[str, Any]:
    """把 CUDA / XPU 的 device properties 归一化成同一个字典。

    XPU 的字段在不同 torch 版本里名字会变，一律 getattr 兜底。
    缺失的字段给保守值（False），宁可判定为"没有能力"也不要误判。
    """
    dev_type = dev_type or backend_type()

    if dev_type == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        return {
            "name": p.name,
            "total_memory_gb": p.total_memory / (1024 ** 3),
            "compute_units": getattr(p, "multi_processor_count", 0),
            "driver": f"cuda{torch.version.cuda}",
            "arch": cap,
            "has_fp16": cap[0] >= 7,
            "has_bf16": cap[0] >= 8,
            "has_xmx": cap[0] >= 8,
            "is_integrated": False,
            "_raw": p,
        }

    if dev_type == "xpu" and accel_available("xpu"):
        p = torch.xpu.get_device_properties(idx)
        # Intel 独显：total_memory 是真实 VRAM
        # Intel 核显：total_memory 是共享内存的一部分，语义不同
        total_mem = float(getattr(p, "total_memory", 0))
        name = str(getattr(p, "name", f"Intel XPU {idx}"))
        # 判断核显/独显
        # 优先用设备的 type 属性（如果有的话）
        dev_type_attr = str(getattr(p, "type", "")).lower()
        if "integrated" in dev_type_attr:
            is_igpu = True
        elif dev_type_attr in ("gpu", "dgpu"):
            is_igpu = False
        else:
            # 回退到名称启发式：先去掉商标符号再判断
            import re as _re
            clean_name = _re.sub(r"\((?:TM|R)\)", "", name)
            _igpu_keys = ("Iris", "UHD")
            _dgpu_keys = ("Arc A", "Arc B", "Arc Pro")
            is_igpu = any(k in clean_name for k in _igpu_keys) and \
                not any(k in clean_name for k in _dgpu_keys)
        return {
            "name": name,
            "total_memory_gb": total_mem / (1024 ** 3),
            "compute_units": getattr(p, "max_compute_units", 0),
            "driver": getattr(p, "driver_version", "unknown"),
            "arch": getattr(p, "architecture", "unknown"),
            "has_fp16": bool(getattr(p, "has_fp16", False)),
            "has_bf16": bool(getattr(p, "has_bfloat16_conversions", False)),
            "has_xmx": bool(
                getattr(p, "has_subgroup_matrix_multiply_accumulate", False)
            ),
            "is_integrated": is_igpu,
            "_raw": p,
        }

    return {
        "name": "CPU",
        "total_memory_gb": 0.0,
        "compute_units": os.cpu_count() or 1,
        "driver": "-",
        "arch": "cpu",
        "has_fp16": False,
        "has_bf16": True,
        "has_xmx": False,
        "is_integrated": False,
        "_raw": None,
    }


def supports_bf16(dev) -> bool:
    t = torch.device(dev)
    if t.type == "cpu":
        return True
    return properties(t.index or 0, t.type).get("has_bf16", False)


def supports_fp16_fast(dev) -> bool:
    """是否值得用 fp16 做加速。

    Intel 核显没有 fp16 加速单元，用了反而更慢 → 返回 False。
    Arc 独显有 XMX，fp16 可用。
    """
    t = torch.device(dev)
    if t.type == "cpu":
        return False
    p = properties(t.index or 0, t.type)
    if t.type == "xpu":
        return bool(p.get("has_xmx") and p.get("has_fp16") and not p.get("is_integrated"))
    return bool(p.get("has_fp16"))


def recommend_dtype(dev) -> torch.dtype:
    """给当前设备推荐推理/训练精度。

    Intel 独显优先 bf16（动态范围大，不需要 GradScaler）；
    核显无 fp16/bf16 加速单元，退回 fp32。
    """
    t = torch.device(dev)
    if t.type == "cpu":
        return torch.float32
    if supports_bf16(dev):
        return torch.bfloat16
    if supports_fp16_fast(dev):
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------- 运行时操作

def empty_cache(dev) -> None:
    t = torch.device(dev).type
    try:
        if t == "cuda":
            torch.cuda.empty_cache()
        elif t == "xpu":
            torch.xpu.empty_cache()
        elif t == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def synchronize(dev) -> None:
    t = torch.device(dev).type
    try:
        if t == "cuda":
            torch.cuda.synchronize()
        elif t == "xpu":
            torch.xpu.synchronize()
        elif t == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def manual_seed_all(seed: int) -> None:
    """同时给 CPU / CUDA / XPU 设种子。任何一项失败都不影响其他。"""
    torch.manual_seed(seed)
    for name in ("cuda", "xpu"):
        try:
            if accel_available(name):
                getattr(torch, name).manual_seed_all(seed)
        except Exception:
            pass


def current_stream(dev):
    t = torch.device(dev).type
    try:
        if t == "cuda":
            return torch.cuda.current_stream(dev)
        if t == "xpu":
            return torch.xpu.current_stream(dev)
    except Exception:
        pass
    return None


def new_event(dev):
    """跨流同步用的 event。"""
    t = torch.device(dev).type
    try:
        if t == "cuda":
            return torch.cuda.Event()
        if t == "xpu":
            return torch.xpu.Event()
    except Exception:
        pass
    return None


def memory_allocated_gb(dev) -> float:
    t = torch.device(dev).type
    try:
        if t == "cuda":
            return torch.cuda.memory_allocated(dev) / (1024 ** 3)
        if t == "xpu":
            return torch.xpu.memory_allocated(dev) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def peak_memory_gb(dev) -> float:
    t = torch.device(dev).type
    try:
        if t == "cuda":
            return torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        if t == "xpu":
            return torch.xpu.max_memory_allocated(dev) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def reset_peak_memory(dev) -> None:
    t = torch.device(dev).type
    try:
        if t == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        if t == "xpu":
            torch.xpu.reset_peak_memory_stats(dev)
    except Exception:
        pass


def memory_report(dev, tag: str = "") -> str:
    """一行显存报告，塞进训练日志用。"""
    return (f"[mem{(' ' + tag) if tag else ''}] "
            f"alloc={memory_allocated_gb(dev):.2f}GB "
            f"peak={peak_memory_gb(dev):.2f}GB")


# ---------------------------------------------------------------- 可见性

def visible_env_name() -> str:
    """子进程隔离显卡用的环境变量名。"""
    if accel_available("xpu") and not accel_available("cuda"):
        return "ZE_AFFINITY_MASK"
    return "_CUDA_VISIBLE_DEVICES"


def set_visible_devices(ids: str) -> None:
    """ids 形如 "0" / "0,1"。替代全项目 11 处硬编码的 _CUDA_VISIBLE_DEVICES。"""
    name = visible_env_name()
    os.environ[name] = ids
    if name == "ZE_AFFINITY_MASK":
        # Level Zero 与 oneAPI 两套选择器都写，兼容不同版本
        os.environ.setdefault(
            "ONEAPI_DEVICE_SELECTOR", f"level_zero:{ids.split(',')[0]}"
        )


# ---------------------------------------------------------------- 调试/兼容

def torch_cuda_shim() -> None:
    """极少数第三方库硬编码 torch.cuda.is_available() 时，可选地打补丁。

    ⚠️ 默认不要启用！它会让"CUDA 专用加速路径"误判为可用，
    进而触发 flash_attn 导入失败。只在确认某依赖库无法配置时打开。
    """
    if accel_available("xpu") and not torch.cuda.is_available():
        warnings.warn(
            "xpu shim active: torch.cuda.is_available() is being faked",
            RuntimeWarning,
            stacklevel=2,
        )
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]


def apply_env_defaults() -> None:
    """在进程启动早期设置推荐环境变量（不覆盖用户已设的值）。"""
    t = backend_type()
    if t == "xpu":
        os.environ.setdefault("PYTORCH_XPU_ALLOC_CONF", "expandable_segments:True")
    elif t == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def describe(dev=None) -> str:
    """给日志用的一行设备描述。"""
    dev = torch.device(dev) if dev is not None else default_device()
    p = properties(dev.index or 0, dev.type)
    if dev.type == "cpu":
        return f"CPU ({p['compute_units']} threads)"
    kind = "iGPU" if p.get("is_integrated") else "dGPU"
    return (f"{p['name']} [{kind}, {p['total_memory_gb']:.1f}GB, "
            f"CU={p['compute_units']}, xmx={p['has_xmx']}, "
            f"bf16={p['has_bf16']}, driver={p['driver']}]")
