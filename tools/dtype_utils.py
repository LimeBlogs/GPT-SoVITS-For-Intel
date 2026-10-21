"""精度（dtype）解析与应用的统一入口。

背景
----
原项目用一个布尔 `is_half` 决定 model.half()。Intel 上要用 bf16，
布尔装不下三态，所以统一改成 dtype。

兼容性
------
`parse()` 同时接受老式的 True/False 和新的 "bf16"/"fp16"/"fp32" 字符串，
所以 WebUI 传 is_half=True 的老代码路径不用改也能继续工作。
"""

from __future__ import annotations

import os

import torch


_TRUE_STRINGS = {"true", "1", "yes", "on", "fp16", "float16", "half", "bf16", "bfloat16"}
_FALSE_STRINGS = {"false", "0", "no", "off", "fp32", "float32", "full"}


def parse(value) -> torch.dtype:
    """把各种表示法解析成 torch.dtype。

    >>> parse(True)      -> torch.float16
    >>> parse("bf16")    -> torch.bfloat16
    >>> parse("False")   -> torch.float32
    >>> parse(None)      -> torch.float32
    """
    if value is None:
        return torch.float32
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, bool):
        return torch.float16 if value else torch.float32
    if isinstance(value, torch.Tensor):
        return value.dtype

    s = str(value).strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "full"):
        return torch.float32
    if s in _TRUE_STRINGS:
        return torch.float16
    if s in _FALSE_STRINGS:
        return torch.float32
    return torch.float32


def from_env(key: str = "is_half", default=None) -> torch.dtype:
    """从环境变量读精度。同时支持 is_half 和 dtype 两个变量名。"""
    for name in (key, "dtype", "GSV_DTYPE"):
        v = os.environ.get(name)
        if v is not None:
            return parse(v)
    return parse(default)


def is_half(dtype: torch.dtype) -> bool:
    """给老代码用：判断是不是低精度。"""
    return dtype in (torch.float16, torch.bfloat16)


def name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype).replace("torch.", ""))


def apply(model, dtype: torch.dtype):
    """替代 model.half()。fp32 时原样返回。"""
    if dtype == torch.float32:
        return model
    return model.to(dtype)


def cast(tensor, dtype: torch.dtype):
    """替代 tensor.half()。只对浮点张量生效。"""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if not tensor.is_floating_point():
        return tensor
    if dtype == torch.float32 and tensor.dtype == torch.float32:
        return tensor
    return tensor.to(dtype)


def resolve(requested, device, *, prefer_bf16: bool = True) -> torch.dtype:
    """综合请求精度与设备能力，给出可用的 dtype。

    规则：
      - 用户明确要 fp32 → fp32（尊重）
      - 设备不支持该精度 → 降级（bf16 → fp16 → fp32）
    """
    from tools import device_utils as du

    want = parse(requested)
    dev = torch.device(device)

    if want == torch.float32 or dev.type == "cpu":
        return torch.float32

    if want == torch.bfloat16 or (prefer_bf16 and want == torch.float16):
        if du.supports_bf16(dev):
            return torch.bfloat16
        if du.supports_fp16_fast(dev):
            return torch.float16
        return torch.float32

    if want == torch.float16:
        return torch.float16 if du.supports_fp16_fast(dev) else torch.float32

    return torch.float32
