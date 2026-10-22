"""P4 补丁：让 PyTorch Lightning 能在 Intel XPU 上训练。

问题
----
pytorch-lightning 2.6.6 的 accelerators 目录只有 cpu / cuda / mps / xla，
没有 ``lightning_fabric.accelerators.xpu``。于是：

* ``Trainer(accelerator="gpu")`` → ``_choose_gpu_accelerator_backend`` 抛
  ``MisconfigurationException: No supported gpu backend found!``
* ``Trainer(accelerator="cpu")`` 能用，但只能跑在 CPU 上，
  ``CPUAccelerator.setup_device`` 会拒绝 ``xpu:0``。

本模块的做法
------------
1. 提供一个最小 ``XPUAccelerator``（对标 ``CUDAAccelerator``，只是把 ``torch.cuda``
   换成 ``torch.xpu``）。
2. 因为 ``_AcceleratorConnector._choose_strategy`` 只对 CUDA/MPS 的 accelerator
   实例特判，自定义 accelerator 会被判成 ``device="cpu"``，所以这里必须同时显式传
   ``strategy=SingleDeviceStrategy(device=torch.device("xpu", 0))``。

注意
----
* 只影响 Lightning 的 device/strategy 装配，不碰模型结构、不改 checkpoint 格式。
* 多卡 DDP 仍走 ``DDPStrategy(process_group_backend="ccl")``，本机单卡用不到。
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - 仅在 Lightning 缺少 XPU 时需要
    from pytorch_lightning.accelerators.accelerator import Accelerator
    from pytorch_lightning.strategies import SingleDeviceStrategy
except Exception as exc:  # pragma: no cover
    raise ImportError("需要 pytorch-lightning >= 2.0 才能使用本垫片") from exc


def xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class XPUAccelerator(Accelerator):
    """最小 XPU accelerator：只实现 device 装配需要的方法。"""

    @staticmethod
    def parse_devices(devices):
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            return [int(d) for d in devices.split(",") if d.strip() != ""]
        return list(devices)

    @staticmethod
    def get_parallel_devices(devices):
        return [torch.device("xpu", int(i)) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count() if xpu_available() else 0

    @staticmethod
    def is_available() -> bool:
        return xpu_available()

    @staticmethod
    def name() -> str:
        return "xpu"

    def setup_device(self, device) -> None:
        torch.xpu.set_device(device)

    def get_device_stats(self, device):
        try:
            return torch.xpu.memory_stats(device)
        except Exception:
            return {}

    def teardown(self) -> None:
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def single_device_kwargs(index: int = 0) -> dict:
    """给 ``pl.Trainer(...)`` 用的 accelerator/strategy/devices 三件套。"""
    return {
        "accelerator": XPUAccelerator(),
        "strategy": SingleDeviceStrategy(device=torch.device("xpu", int(index))),
        "devices": 1,
    }


def lightning_has_xpu() -> bool:
    """Lightning 自带 XPU accelerator 时为 True（2.0~2.2 某些版本 / Intel 分支）。"""
    try:
        from lightning_fabric.accelerators.xpu import XPUAccelerator as _Official  # noqa: F401

        return True
    except Exception:
        return False
