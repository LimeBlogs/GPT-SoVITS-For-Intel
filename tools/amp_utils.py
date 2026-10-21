"""统一的 AMP（自动混合精度）辅助。

原项目用 torch.cuda.amp.autocast / GradScaler，在 XPU 上不适用。
这里做一层归一化，调用方只传 device，不关心后端。

Intel 独显要点
--------------
1. Arc 的 XMX 支持 bf16，且 bf16 不需要梯度缩放（没有 fp16 的溢出问题）。
   → GradScaler 在 XPU 上直接禁用，比"开着但 working"更稳。
2. mel loss / STFT / 判别器一律 force_fp32，不要扔进 autocast。
3. autocast 对 CPU 无意义，返回 nullcontext。
"""

from __future__ import annotations

import contextlib
import logging

import torch

from tools import device_utils as du

_log = logging.getLogger(__name__)


def autocast_ctx(device, enabled: bool, dtype=None, force_fp32: bool = False):
    """统一的 autocast 上下文。

    参数
    ----
    device : str | torch.device
    enabled : bool
        总开关，一般来自 hps.train.fp16_run 或配置项。
    dtype : torch.dtype | None
        显式指定精度；None 时按后端自动选（XPU→bf16，CUDA→fp16）。
    force_fp32 : bool
        True 时永远返回 nullcontext，用于 mel loss / 判别器 / STFT。

    返回
    ----
    一个可 with 的 context manager，永不抛异常。
    """
    if force_fp32 or not enabled:
        return contextlib.nullcontext()

    dev_type = torch.device(device).type
    if dev_type == "cpu":
        return contextlib.nullcontext()

    if dtype is None:
        dtype = torch.bfloat16 if dev_type == "xpu" else torch.float16

    try:
        return torch.amp.autocast(device_type=dev_type, dtype=dtype)
    except (TypeError, RuntimeError, AttributeError) as e:
        _log.warning("autocast unavailable (%s), falling back to fp32: %s", dev_type, e)
        return contextlib.nullcontext()


def make_scaler(device, enabled: bool):
    """GradScaler 工厂。

    - XPU 上一律禁用（bf16 不需要缩放）
    - CUDA 上沿用原行为
    - 任何异常都回落到 _NullScaler，保证训练流程不中断
    """
    dev_type = torch.device(device).type

    if dev_type == "xpu":
        # 不尝试启用：Intel 上 bf16 缩放反而引入不稳定
        try:
            return torch.amp.GradScaler("xpu", enabled=False)
        except (TypeError, RuntimeError, AttributeError):
            return _NullScaler()

    if dev_type == "cpu":
        return _NullScaler()

    try:
        return torch.amp.GradScaler(dev_type, enabled=enabled)
    except (TypeError, RuntimeError, AttributeError):
        try:
            # 老版本 torch 只接受关键字
            from torch.cuda.amp import GradScaler as _Legacy
            return _Legacy(enabled=enabled)
        except Exception:
            return _NullScaler()


class _NullScaler:
    """scaler 的空实现：所有方法都是恒等操作。

    用于 CPU / XPU / 老版本 torch，让训练代码不需要写 if。
    """

    def scale(self, x):
        return x

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer, *a, **kw):
        return optimizer.step(*a, **kw)

    def update(self, *a, **kw):
        return None

    def get_scale(self):
        return 1.0

    def state_dict(self):
        return {}

    def load_state_dict(self, *a, **kw):
        return None


# ---------------------------------------------------------------- 精度策略

def selective_fp32(module, keep_fp32_names=("enc_q", "flow")):
    """把模块主体降到 bf16，但指定子模块保持 fp32。

    用于 SoVITS 生成器：flow 的 log_scale 在 bf16 下精度不足会出金属音。
    """
    for name in keep_fp32_names:
        sub = getattr(module, name, None)
        if sub is not None and hasattr(sub, "float"):
            sub.float()
    return module


def cast_module(module, dtype: torch.dtype):
    if dtype == torch.float32:
        return module
    return module.to(dtype)


def cast_batch(batch, dtype: torch.dtype, device):
    """把 batch 里的张量统一转到 device + dtype。

    只处理 Tensor 和 list/tuple，其他原样返回。
    """
    def _one(x):
        if isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                return x.to(device, non_blocking=True)
            return x.to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(_one(i) for i in x)
        return x

    if isinstance(batch, (list, tuple)):
        return type(batch)(_one(x) for x in batch)
    return _one(batch)


# ---------------------------------------------------------------- 有限性检查

def all_finite(*tensors) -> bool:
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if not torch.isfinite(t.detach()).all():
            return False
    return True


def check_finite_and_dump(losses: dict, callback=None) -> None:
    """loss 出现 NaN/Inf 时调用 callback（通常是保存现场 + 退出）。

    losses 形如 {"loss_disc": t, "loss_gen": t, ...}
    """
    bad = {k: float(v) for k, v in losses.items()
           if isinstance(v, torch.Tensor) and not torch.isfinite(v.detach()).all()}
    if bad:
        _log.error("non-finite loss detected: %s", bad)
        if callback is not None:
            callback(bad)
        return
