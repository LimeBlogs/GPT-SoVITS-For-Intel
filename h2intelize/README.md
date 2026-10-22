# INDEX · GPT-SoVITS → Intel Arc A750 移植规格（v2）

四份独立文档，每份可单独交给 AI 执行。**已按 A750 实测探针结果定案。**

---

## 已确定的环境与结论

```
硬件  : Intel Arc A750, 8 GB (实测报 7.8 GB)
        max_compute_units = 448
        driver = 1.15.37858
        has_fp16 = True
        has_bfloat16_conversions = False
        has_subgroup_matrix_multiply_accumulate = False
系统  : Windows 11
Python: 3.12（venv 在 D:\GPSVT\GPT-SoVITS\.venv）
torch : 2.7.1+xpu
仓库  : D:\GPSVT\GPT-SoVITS
Shell : PowerShell 7
```

### 三条已定案的决策

| # | 决策 | 依据 |
|---|---|---|
| **1** | **全程 fp32，不用 bf16** | bf16 无收益（matmul 0.37 vs 0.35 ms），GAN step 反而慢 41%；且 STFT 在 bf16 下直接报错（MKL FFT 不支持） |
| **2** | **不做 STFT 兜底** | fp32 下 STFT 反传正常（8.43 ms），比 CPU 还快 |
| **3** | **不做 bf16 GAN 稳定性调参** | 不用 bf16 就不存在这个问题 |

**因此 P4 从 6 天缩到 3 天，总工期 12.6 天 → 9.5 天。**

---

## 文档

| 文档 | 前置 | 工期 | 完成后 |
|---|---|---|---|
| [P1-基础层.md](./P1-基础层.md) | 无 | 4 天 | 推理能跑（fp32），无 CUDA 硬编码 |
| [P2-ONNX子链.md](./P2-ONNX子链.md) | P1 | 2 天 | G2PW / UVR5 走对后端 |
| [P3-推理加速.md](./P3-推理加速.md) | P1 | 0.5 天（P3a） | `gpt_streaming` 可用 |
| [P4-训练移植.md](./P4-训练移植.md) | P1 | **3 天** | s2 / s1 训练跑通 |
| [P0-探针结果与决策.md](./P0-探针结果与决策.md) | — | — | 实测数据与决策依据（参考） |

---

## 依赖关系

```
P1 ──┬──► P2   （互不相关，可并行）
     ├──► P3a  （互不相关，可并行）
     └──► P4   （最重要）
```

**P2 / P3a 都不阻塞 P4。** 想尽快能训练：P1 → 直接 P4。

---

## 探针结果速查

| 项 | fp32 | 结论 |
|---|---|---|
| STFT 反传 | ✅ 8.43 ms | 可用（内部退 CPU） |
| 完整 mel loss | ✅ 9.33 ms | 可用 |
| GAN mini step | ✅ 6.88 ms | 可用 |
| GRU | ✅ 187 ms | 可用但慢，**仅 v2Pro 路径用到** |
| ConvTranspose1d | ✅ 10.60 ms | 可用 |
| DataLoader pin | ✅ 1.83 ms | 可用 |
| oneCCL | ❌ | 单卡不需要 |
| bf16 STFT | ❌ MKL FFT 不支持 | **所以走 fp32** |

---

## 三个最容易踩的坑

| # | 坑 | 在哪 |
|---|---|---|
| 1 | `from tools import` 缺路径垫片 → ImportError | P1 §R1（P2/P4 各自重写） |
| 2 | **Windows `num_workers=4` → worker 必崩** | **P4 §R3** |
| 3 | `hps.train.get()` → AttributeError | P4 §R2 |

---

## 建议执行顺序

```
Day 0     tools/_selftest.py（已跑过探针，环境就绪）
Day 1-4   P1
Day 5-7   P4          ← 最重要，缩到 3 天
Day 8     P3a
Day 9-10  P2
之后      P3b（可选，需先跑 xpu_graph_probe.py）
```

---

## 通用铁律（四份文档都适用）

1. **路径垫片**：`GPT_SoVITS/xxx.py` → `parents[1]`；`GPT_SoVITS/sub/xxx.py` → `parents[2]`
2. **`getattr` 兜底**：`hps` 是 `HParams`，没有 `.get()`
3. **不许伪报**：没跑过的验收项标 `TODO(human)`
4. **一个 Task 一个 commit**
5. **精度一律 fp32**，不要引入 bf16

---

## 已有的配套文件

| 文件 | 用途 |
|---|---|
| `tools/device_utils.py` | 360 行，设备访问唯一入口 |
| `tools/amp_utils.py` | 179 行，autocast / scaler（fp32 下多数是空操作） |
| `tools/dtype_utils.py` | 121 行，精度解析 |
| `tools/_selftest.py` | 192 行，60 项自检 |
| `xpu_train_probe.py` | 已跑完，结果见 P0 文档 |
| `xpu_graph_probe.py` | P3b 前再跑 |
| `requirements-xpu-win.txt` | 纯 ASCII，XPU 适配版依赖 |
| `install_xpu_win.ps1` | 一键安装 |
| `set_caches.ps1` | 缓存挪出 C 盘 |
