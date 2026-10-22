# install_xpu_win.ps1
# ---------------------------------------------------------------------------
# GPT-SoVITS · Intel Arc · Windows 11 · venv 一键安装
#
# 用法（在仓库根目录执行）：
#   powershell -ExecutionPolicy Bypass -File install_xpu_win.ps1
#
# 可选参数：
#   -SkipVenv        不创建 venv，用当前环境
#   -PythonExe       python 可执行文件路径（默认 py -3.12）
#   -Mirror          pypi 镜像（默认 pku）
#
# 说明：
#   - 分阶段安装，每步失败会停下并告诉你怎么修
#   - 不安装 flash-attn（不支持 XPU）
#   - 不安装 onnxruntime-gpu（Intel 上无效）
# ---------------------------------------------------------------------------

param(
    [switch]$SkipVenv,
    [string]$PythonExe = "",
    [string]$Mirror = "https://mirrors.pku.edu.cn/pypi/simple"
)

$ErrorActionPreference = "Stop"
$ROOT = Get-Location

function Step($n, $msg) {
    Write-Host ""
    Write-Host "==== [$n] $msg ====" -ForegroundColor Cyan
}

function Die($msg) {
    Write-Host ""
    Write-Host "✗ $msg" -ForegroundColor Red
    Write-Host "  上下文：$($MyInvocation.PSCommandPath)" -ForegroundColor DarkGray
    exit 1
}

# ---------------------------------------------------------------- 0. 环境检查

Step 0 "环境检查"

if (-not (Test-Path "requirements-xpu-win.txt")) {
    Die "找不到 requirements-xpu-win.txt，请把脚本放在仓库根目录运行"
}

if ([string]::IsNullOrEmpty($PythonExe)) {
    # 优先 py launcher 指定 3.12
    try {
        $v = & py -3.12 -c "import sys; print(sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0) { $PythonExe = "py -3.12" }
    } catch { }
    if ([string]::IsNullOrEmpty($PythonExe)) {
        $PythonExe = "python"
    }
}

Write-Host "Python : $PythonExe"
& cmd /c "$PythonExe -c ""import sys; print(sys.version)"""
if ($LASTEXITCODE -ne 0) { Die "python 不可用，请指定 -PythonExe" }

$pyVer = & cmd /c "$PythonExe -c ""import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"""
Write-Host "版本   : $pyVer"
if ($pyVer -ne "3.12") {
    Write-Host ""
    Write-Host "⚠ 当前不是 Python 3.12。" -ForegroundColor Yellow
    Write-Host "  requirements-xpu-win.txt 里 pyopenjtalk / jieba_fast 用的是 cp312 预编译 wheel。" -ForegroundColor Yellow
    Write-Host "  非 3.12 会退化成源码编译，需要 MSVC + CMake，很可能失败。" -ForegroundColor Yellow
    $ans = Read-Host "  继续？(y/N)"
    if ($ans -ne "y") { exit 1 }
}

# 显卡检查
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Intel.*Arc" }
if ($gpu) {
    Write-Host "显卡   : $($gpu.Name)  driver=$($gpu.DriverVersion)" -ForegroundColor Green
} else {
    Write-Host "显卡   : 未检测到 Intel Arc（可能只是警告）" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 1. venv

if (-not $SkipVenv) {
    Step 1 "创建 venv"
    if (Test-Path ".venv") {
        Write-Host "  .venv 已存在，复用"
    } else {
        & cmd /c "$PythonExe -m venv .venv"
        if ($LASTEXITCODE -ne 0) { Die "venv 创建失败" }
    }
    $PY = Join-Path $ROOT ".venv\Scripts\python.exe"
    if (-not (Test-Path $PY)) { Die "找不到 $PY" }
} else {
    Step 1 "使用当前环境"
    $PY = (Get-Command python).Source
}

Write-Host "解释器 : $PY"

# ---------------------------------------------------------------- 2. 升级 pip

Step 2 "升级 pip / setuptools / wheel"
& $PY -m pip install -U pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Die "pip 升级失败（检查网络 / 代理）" }

# ---------------------------------------------------------------- 3. torch XPU

Step 3 "安装 PyTorch XPU 版（约 2.5 GB，会慢）"
Write-Host "  这一步如果失败，最常见的两个原因："
Write-Host "   a) 网络到 download.pytorch.org 不通 → 用代理或稍后重试"
Write-Host "   b) 装成了 CUDA/CPU 版 → 会在这里被下一步的验证拦住"
Write-Host ""

& $PY -m pip install torch==2.7.1 torchaudio==2.7.1 `
    --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) {
    Die "torch XPU 安装失败。若网络问题，可先设置代理：`$env:HTTPS_PROXY='http://127.0.0.1:7890'"
}

# ---------------------------------------------------------------- 4. 验证 torch

Step 4 "验证 torch 是否为 XPU 版"
$check = @"
import sys
import torch
print('torch        :', torch.__version__)
print('xpu build    :', getattr(torch.version, 'xpu', 'NO XPU BUILD'))
print('xpu available:', hasattr(torch, 'xpu') and torch.xpu.is_available())
if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
    sys.exit(1)
p = torch.xpu.get_device_properties(0)
print('device       :', getattr(p, 'name', '?'))
print('vram (GB)    :', round(getattr(p, 'total_memory', 0) / 1024**3, 1))
print('compute units:', getattr(p, 'max_compute_units', '?'))
a = torch.randn(512, 512, device='xpu', dtype=torch.bfloat16)
print('bf16 matmul  :', 'OK' if (a @ a) is not None else 'FAIL')
"@
$check | Out-File -Encoding utf8 _check_torch.py
& $PY _check_torch.py
$ok = $LASTEXITCODE -eq 0
Remove-Item _check_torch.py -ErrorAction SilentlyContinue

if (-not $ok) {
    Write-Host ""
    Write-Host "✗ torch 装错了（不是 XPU 版，或 XPU 不可用）" -ForegroundColor Red
    Write-Host ""
    Write-Host "排查步骤：" -ForegroundColor Yellow
    Write-Host "  1. 是不是装的 CPU/CUDA 版？" -ForegroundColor Yellow
    Write-Host "     $PY -c ""import torch; print(torch.version.xpu)""" -ForegroundColor Yellow
    Write-Host "     输出 None 或报错 → 装错包了" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  2. 卸载重装：" -ForegroundColor Yellow
    Write-Host "     $PY -m pip uninstall -y torch torchaudio" -ForegroundColor Yellow
    Write-Host "     $PY -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/xpu" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  3. Arc 驱动装了吗？(不是 Windows Update 版本)" -ForegroundColor Yellow
    Write-Host "     https://www.intel.com/content/www/us/en/download/785597/" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ torch XPU 就绪" -ForegroundColor Green

# ---------------------------------------------------------------- 5. 项目依赖

Step 5 "安装项目依赖（约 1.5 GB）"
& $PY -m pip install -r requirements-xpu-win.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠ 依赖安装有失败项，常见原因：" -ForegroundColor Yellow
    Write-Host "  · opencc 源码编译 → 本文件已用预编译版，不应出现" -ForegroundColor Yellow
    Write-Host "  · numba / llvmlite 版本不匹配 → 试 pip install numba --upgrade" -ForegroundColor Yellow
    Write-Host "  · 网络超时 → 重跑本步即可（pip 会跳过已装好的）" -ForegroundColor Yellow
    Write-Host ""
    $ans = Read-Host "  继续做后续验证？(y/N)"
    if ($ans -ne "y") { exit 1 }
}

# ---------------------------------------------------------------- 6. IPEX（可选）

Step 6 "安装 IPEX（可选，只影响推理加速）"
& $PY -m pip install intel-extension-for-pytorch `
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ IPEX 安装失败 —— 不影响主流程，代码里已做 try/except 兜底" -ForegroundColor Yellow
} else {
    Write-Host "✓ IPEX 就绪" -ForegroundColor Green
}

# ---------------------------------------------------------------- 7. 清理冲突包

Step 7 "清理会冲突的 ONNX 包"
foreach ($p in @("onnxruntime-gpu", "onnxruntime-openvino", "onnxruntime")) {
    & $PY -m pip uninstall -y $p 2>$null | Out-Null
}
& $PY -m pip install "onnxruntime-directml>=1.19,<2"
if ($LASTEXITCODE -ne 0) { Die "onnxruntime-directml 安装失败" }

# ---------------------------------------------------------------- 8. 最终验证

Step 8 "最终验证"

$final = @"
import sys
ok = True

def line(name, fn):
    global ok
    try:
        r = fn()
        print(f'  OK   {name:38s} {r}')
    except Exception as e:
        ok = False
        print(f'  FAIL {name:38s} {type(e).__name__}: {e}')

print('--- torch ---')
line('xpu available', lambda: __import__('torch').xpu.is_available())

print('--- onnxruntime ---')
def _ort():
    import onnxruntime as ort
    p = ort.get_available_providers()
    assert 'DmlExecutionProvider' in p, f'no DML: {p}'
    return p
line('providers (need DML)', _ort)

print('--- 关键依赖 ---')
for m in ['numpy', 'scipy', 'librosa', 'numba', 'gradio', 'pytorch_lightning',
          'transformers', 'peft', 'onnxruntime', 'funasr', 'modelscope',
          'opencc', 'pyopenjtalk', 'jieba_fast', 'ctranslate2', 'av']:
    line(m, lambda m=m: getattr(__import__(m), '__version__', 'n/a'))

sys.exit(0 if ok else 1)
"@
$final | Out-File -Encoding utf8 _final_check.py
& $PY _final_check.py
$finalOk = $LASTEXITCODE -eq 0
Remove-Item _final_check.py -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 9. 汇总

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
if ($finalOk) {
    Write-Host " ✓ 安装完成" -ForegroundColor Green
} else {
    Write-Host " ⚠ 安装完成，但有失败项（见上）" -ForegroundColor Yellow
}
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "接下来：" -ForegroundColor Cyan
Write-Host "  1. 跑基础自检："
Write-Host "       .\.venv\Scripts\python.exe tools\_selftest.py"
Write-Host "  2. 跑训练探针（决定 P4 要不要做 STFT 兜底）："
Write-Host "       .\.venv\Scripts\python.exe xpu_train_probe.py"
Write-Host "  3. 设环境变量："
Write-Host "       `$env:GSV_DEVICE='xpu'"
Write-Host "  4. 启动 WebUI："
Write-Host "       .\.venv\Scripts\python.exe webui.py zh_CN"
Write-Host ""
Write-Host "注意：" -ForegroundColor Yellow
Write-Host "  · 本机如果有 ffmpeg，请把 ffmpeg.exe / ffprobe.exe 放到仓库根目录" -ForegroundColor Yellow
Write-Host "  · 建议把整个仓库目录加入 Windows Defender 排除列表（否则存 checkpoint 很慢）" -ForegroundColor Yellow
