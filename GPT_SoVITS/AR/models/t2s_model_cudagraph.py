"""兼容层：老 import 路径保持不变，内部按需加载。

原来的顶层 `import flash_attn` 在 XPU 环境会直接 ImportError，
因为 flash_attn 包不支持 Intel 后端。改成模块级 __getattr__ 惰性加载。
"""


def __getattr__(name):
    if name == "CUDAGraphRunner":
        from GPT_SoVITS.Accel.PyTorch.AR.t2s_engine import T2SEngine

        return T2SEngine

    table = {}

    if name in ("T2SRequest", "T2SResult"):
        from GPT_SoVITS.Accel import T2SRequest, T2SResult

        table = {"T2SRequest": T2SRequest, "T2SResult": T2SResult}
        return table[name]

    from GPT_SoVITS.Accel.PyTorch.AR.backends.flash_attn_varlen_cuda_graph import (
        Attention,
        T2SDecoder,
        TransformerBlock,
        TransformerDecoder,
    )

    table = {
        "Attention": Attention,
        "T2SDecoder": T2SDecoder,
        "TransformerBlock": TransformerBlock,
        "TransformerDecoder": TransformerDecoder,
    }
    if name not in table:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return table[name]
