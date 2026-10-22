"""XPU eager 后端：复用 PyTorch 通用 SDPA 实现（完全设备无关）。

torch_varlen.py 用的是 F.scaled_dot_product_attention，
没有 CUDA 依赖，可直接复用。
"""

from ....PyTorch.AR.backends.torch_varlen import (
    Attention,
    TransformerBlock,
    TransformerDecoder,
    T2SDecoder,
)

__all__ = ["Attention", "TransformerBlock", "TransformerDecoder", "T2SDecoder"]
