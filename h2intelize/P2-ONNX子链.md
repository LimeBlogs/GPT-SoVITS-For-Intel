# P2 · ONNX 子链（中文多音字 / 人声分离 / 语音识别）

> **独立执行单元。**
> 前置：P1 已完成
> 工期：2 天
> 完成后：**G2PW / UVR5 走 DirectML，ASR 跑 CPU int8**

---

# 0. 环境

```
OS   : Windows 11
GPU  : Intel Arc A750 8GB
Shell: PowerShell 7
仓库 : D:\GPSVT\GPT-SoVITS
venv : .venv
```

---

# 1. 目标与现状

## 1.1 问题

`requirements.txt` 在 x86_64 上装的是 **`onnxruntime-gpu`（CUDA 版）**，Intel 机器没有 CUDA，于是：

```python
# text/g2pw/onnx_api.py:343
if "CUDAExecutionProvider" in onnxruntime.get_available_providers():   # False
    ... GPU
else:
    ... CPU          # ← 静默退 CPU，无报错
```

**表现：中文多音字处理很慢，且没有任何提示。**

## 1.2 完成定义（DoD）

| # | 验收项 | 期望 |
|---|---|---|
| D1 | provider 列表含 `DmlExecutionProvider` | 见 §6.1 |
| D2 | G2PW 能 import 并加载 | 见 §6.2 |
| D3 | UVR5 走 DML | 见 §6.3 |
| D4 | ASR 出文字 | 见 §6.4 |
| D5 | 无 CUDA provider 硬编码 | 见 §6.5 |

---

# 2. 硬约束

## R1：不要同时装多个 onnxruntime 变体

`onnxruntime` / `onnxruntime-gpu` / `onnxruntime-openvino` / `onnxruntime-directml` **共用同一个 `onnxruntime` 命名空间**，装两个会互相覆盖，出现随机 ImportError。

Windows 上**只装 `onnxruntime-directml`**。

## R2：G2PW 默认留 CPU

`g2pW.onnx` 是小模型、单句输入，DirectML 的固定提交开销会让它**变慢**（详见附录 A）。所以设计成开关，默认 CPU。

## R3：一个 Task 一个 commit

## R4：不许伪报

`get_available_providers()` 里没有 `DmlExecutionProvider` 就说没成功，不要写"已生效"。

---

# 3. 已验证事实

| # | 事实 |
|---|---|
| F1 | `requirements.txt` 在 x86_64 上装 `onnxruntime-gpu` |
| F2 | `g2pw/onnx_api.py:343-353` 只认 CUDA，否则纯 CPU |
| F3 | `mdxnet.py:85-91` 的 provider 列表**已含 DML**（CUDA → DML → CPU），装完即生效 |
| F4 | `g2pw/onnx_api.py:22-23` 有 `onnxruntime.preload_dlls()`，`onnxruntime-directml` 不一定有这个函数 |
| F5 | `faster-whisper` 底层是 `ctranslate2`，**没有 Intel GPU 后端** |
| F6 | `funasr_asr.py:58` 走 PyTorch，理论上能上 XPU |

---

# 4. 任务清单

```
T2-1  装 DirectML 版 ORT，卸掉冲突包
T2-2  新建 tools/ort_providers.py
T2-3  改 g2pw/onnx_api.py
T2-4  改 uvr5/mdxnet.py
T2-5  改 ASR 两个脚本
T2-6  验收
```

---

## T2-1 安装

```powershell
cd D:\GPSVT\GPT-SoVITS
.\.venv\Scripts\Activate.ps1

# 卸掉所有冲突变体
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino onnxruntime-directml

# 只装 DirectML 版
pip install "onnxruntime-directml>=1.19,<2"

# 验证
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

期望输出含 `DmlExecutionProvider` 和 `CPUExecutionProvider`。

**若只有 `CPUExecutionProvider`**：说明装的是普通版，重跑上面的卸载 + 安装。

---

## T2-2 新建 `tools/ort_providers.py`

```python
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
```

**验收**：

```powershell
.\.venv\Scripts\python.exe -c @"
from tools.ort_providers import select_providers, select_providers_for_g2pw, describe
print(describe())
print('default :', select_providers())
print('for g2pw:', select_providers_for_g2pw())
"@
```

期望 `default` 首个 provider 是 `DmlExecutionProvider`，`for g2pw` 是 `CPUExecutionProvider`。

---

## T2-3 改 `GPT_SoVITS/text/g2pw/onnx_api.py`

### 3a. 顶部加 import（**注意是 3 层**）

```python
import sys
import pathlib

# GPT_SoVITS/text/g2pw/onnx_api.py → 向上 3 层到仓库根
_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.ort_providers import select_providers_for_g2pw
```

### 3b. 修 `preload_dlls` 守卫（原 L22-23）

```python
# 原
if os.name == "nt":
    onnxruntime.preload_dlls()

# 新
if os.name == "nt" and hasattr(onnxruntime, "preload_dlls"):
    try:
        onnxruntime.preload_dlls()
    except Exception:
        pass
```

> `onnxruntime-directml` 不一定有 `preload_dlls`，不守卫会 AttributeError。

### 3c. 替换 provider 选择（原 L343-353）

```python
# 原
if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
    self.session_g2pw = onnxruntime.InferenceSession(
        onnx_path, sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
else:
    self.session_g2pw = onnxruntime.InferenceSession(
        onnx_path, sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

# 新
providers, provider_options = select_providers_for_g2pw()
self.session_g2pw = onnxruntime.InferenceSession(
    onnx_path,
    sess_options=sess_options,
    providers=providers,
    provider_options=provider_options,
)
```

**验收**：

```powershell
.\.venv\Scripts\python.exe -c @"
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
sys.path.insert(0, str(pathlib.Path('GPT_SoVITS').resolve()))
from text.g2pw.onnx_api import G2PWConverter
print('G2PW import OK')
"@
```

报的不是 `ImportError` / `AttributeError` 就算过。

---

## T2-4 改 `tools/uvr5/mdxnet.py`

原 L84-92：

```python
self.model = ort.InferenceSession(
    os.path.join(args.onnx, self.model_.target_name + ".onnx"),
    providers=[
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

改成：

```python
from tools.ort_providers import select_providers

providers, provider_options = select_providers()
self.model = ort.InferenceSession(
    os.path.join(args.onnx, self.model_.target_name + ".onnx"),
    providers=providers,
    provider_options=provider_options,
)
```

> `tools/uvr5/mdxnet.py` 在 `tools/` 下，若报 `ModuleNotFoundError: tools`，加垫片：
> ```python
> _ROOT = pathlib.Path(__file__).resolve().parents[2]
> if str(_ROOT) not in sys.path:
>     sys.path.insert(0, str(_ROOT))
> ```

---

## T2-5 改 ASR

### 5a. `tools/asr/fasterwhisper_asr.py`（L108）

`ctranslate2` **没有 Intel GPU 后端**，只能退 CPU：

```python
# 原
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_path, device=device, compute_type=precision)

# 新
dev_type = torch.device(du.default_device()).type
if dev_type == "cuda":
    device, ctype = "cuda", precision
else:
    device, ctype = "cpu", "int8"        # ctranslate2 无 Intel GPU 后端
model = WhisperModel(model_path, device=device, compute_type=ctype)
```

### 5b. `tools/asr/funasr_asr.py`（L58）

```python
# 原
device = "cuda" if torch.cuda.is_available() else "cpu"
# 新
device = str(du.default_device())
```

> FunASR 内部可能有 `.cuda()` 硬编码。若报 `RuntimeError: ... cuda`，退 `device = "cpu"`，在 commit message 写明。

---

## T2-6 验收

见 §6。

---

# 5. 禁止事项

| # | 禁止 |
|---|---|
| X1 | 同时装多个 onnxruntime 变体（R1） |
| X2 | 装 `onnxruntime-gpu` |
| X3 | 把 G2PW 默认改成 DML（小 batch 会变慢） |
| X4 | 改 `mdxnet.py` 的 provider **优先级**（CUDA 优先对 CUDA 机器是正确的） |
| X5 | 强行让 faster-whisper 走 GPU（没有后端） |
| X6 | 伪报 DML 生效 |

---

# 6. 验收命令（PowerShell）

## 6.1 D1：provider 列表

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 6.2 D2：G2PW 加载

见 §T2-3 的验收命令。

## 6.3 D3：UVR5 走 DML

```powershell
.\.venv\Scripts\python.exe tools\uvr5\webui.py "0" 9873
```

跑一次人声分离，看控制台里 `mdxnet.py:80` 附近的 `ort.get_available_providers()` 输出，首个应为 DML。

**通过标准**：产出人声 + 伴奏两个文件。

## 6.4 D4：ASR

```powershell
.\.venv\Scripts\python.exe tools\asr\fasterwhisper_asr.py -i <某个wav> -o out
```

通过标准：产出文本，日志显示 `device=cpu, compute_type=int8`。

## 6.5 D5：无硬编码 CUDA provider

```powershell
Get-ChildItem tools, GPT_SoVITS -Recurse -Filter *.py |
  Select-String -Pattern 'CUDAExecutionProvider' |
  Select-Object Path, LineNumber
```

期望只在 `tools/ort_providers.py` 里出现。

---

# 7. 停止条件

| 情况 | 动作 |
|---|---|
| provider 列表一直只有 CPU | 停，报告 `pip list \| findstr onnx` |
| G2PW 报 `AttributeError: preload_dlls` | 停，§T2-3b 守卫没加对 |
| UVR5 报 DML 初始化失败 | 停，报完整 traceback（可能驱动 / DX12 问题） |
| FunASR 报 cuda 错误 | 退 CPU，记录原因 |
| 任何验收改 3 次不过 | 停，报告 |

---

# 8. 交付

| 文件 | 类型 |
|---|---|
| `tools/ort_providers.py` | 新建 |
| `GPT_SoVITS/text/g2pw/onnx_api.py` | 改（3 处） |
| `tools/uvr5/mdxnet.py` | 改（1 处） |
| `tools/asr/fasterwhisper_asr.py` | 改（1 处） |
| `tools/asr/funasr_asr.py` | 改（1 处） |

完成后不阻塞任何阶段。

---

# 附录 A：为什么 G2PW 留 CPU

`g2pW.onnx` 输入是「一句话的字符序列」，典型形状 `[1, ~30]`。

| 操作 | CPU | DML |
|---|---|---|
| 小 batch（1×30） | ~2 ms | ~8-15 ms（含提交/同步开销） |
| 大 batch（64×512） | ~40 ms | ~5 ms |

G2PW 永远是小 batch → CPU 更快。
UVR5 是频谱图上的大张量卷积 → DML 收益明显（3 分钟音频约 90s → 25s）。

**所以按 module 分别选，不是"一律上 GPU"。**

---

# 附录 B：`GSV_ORT_PROVIDER` 用法

```powershell
# 强制 CPU（对比性能）
$env:GSV_ORT_PROVIDER = "CPUExecutionProvider"

# 强制 DML
$env:GSV_ORT_PROVIDER = "DmlExecutionProvider"

# G2PW 试 DML（默认 CPU）
$env:GSV_G2PW_CPU = "0"
```
