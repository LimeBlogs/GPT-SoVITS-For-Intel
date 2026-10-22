"""XPU eager 后端（解析别名）。

``T2SEngine.load_decoder`` 用 ``import_module(f".backends.{backend}", package=__package__)``
解析后端，而 ``__package__`` 固定为 ``GPT_SoVITS.Accel.PyTorch.AR``；同样
``create_acceleration`` 也拼 ``GPT_SoVITS.Accel.PyTorch.AR.backends.<backend>``。
因此后端模块必须能在本目录下被找到，否则 A2/A3 会在加载权重时直接 ImportError。

真正的实现放在 ``GPT_SoVITS.Accel.XPU.AR.backends.sdpa_varlen_xpu``，
这里只做转发，保持 XPU 相关代码集中在 ``Accel/XPU`` 下。
"""

from ....XPU.AR.backends.sdpa_varlen_xpu import (
    Attention,
    T2SDecoder,
    TransformerBlock,
    TransformerDecoder,
)

__all__ = ["Attention", "TransformerBlock", "TransformerDecoder", "T2SDecoder"]
