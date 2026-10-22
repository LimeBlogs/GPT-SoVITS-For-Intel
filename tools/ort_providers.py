"""ONNX Runtime 执行器选择，全项目共用。

Windows + Intel：主力是 DirectML。
CUDA 机器上行为与原来一致（优先 CUDA）。
"""

import os

import onnxruntime


# G2PW 默认走 CPU（小模型 + 小 batch，DML 反而慢）
G2PW_PREFER_CPU = os.environ.get("GSV_G2PW_CPU", "1") == "1"


def select_providers(prefer: str | None = None):
    """返回 (providers, provider_options)。

    顺序：CUDA > DirectML > OpenVINO > CPU
    GSV_ORT_PROVIDER 可强制指定其一。
    """
    avail = set(onnxruntime.get_available_providers())
    forced = prefer or os.environ.get("GSV_ORT_PROVIDER")

    table = [
        ("CUDAExecutionProvider", []),
        ("DmlExecutionProvider", [{"device_id": 0}]),
        ("OpenVINOExecutionProvider", [{
            "device_type": os.environ.get("GSV_OV_DEVICE", "GPU"),
            "precision": os.environ.get("GSV_OV_PRECISION", "FP16"),
            "num_of_threads": "0",
            "cache_dir": os.environ.get("GSV_OV_CACHE", ""),
        }]),
    ]

    if forced:
        table = [t for t in table if t[0] == forced] or table

    for name, opts in table:
        if name in avail:
            return [name, "CPUExecutionProvider"], opts + [{}]
    return ["CPUExecutionProvider"], [{}]


def select_providers_for_g2pw():
    """G2PW 专用：默认 CPU，GSV_G2PW_CPU=0 时才用 GPU。"""
    if G2PW_PREFER_CPU:
        return ["CPUExecutionProvider"], [{}]
    return select_providers()


def describe() -> str:
    return f"ort={onnxruntime.__version__} providers={onnxruntime.get_available_providers()}"
