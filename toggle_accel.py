#!/usr/bin/env python3
"""开关 P3a 的 XPU 推理后端（出问题时用来快速回滚）

背景
----
P3a 之后 `resolve_acceleration('xpu', ...)` 会返回 "sdpa_varlen_xpu"，
于是**所有**推理都改走 Accel 的 decoder（不只是 gpt_streaming）。

如果这条新路径在驱动层崩了（比如 AssertHandler::printMessage 这种
Level Zero 断言，进程直接 abort，前端只看到 Connection errored out），
最快的恢复方式是把后端模块挪走 —— create_acceleration() 会 ImportError，
被 except 吞掉后返回 None，推理自动退回原版 infer_panel 路径。

代价：gpt_streaming（语义流式）不可用。普通推理不受影响。

用法
----
    cd D:\\GPSVT\\GPT-SoVITS
    .\\.venv\\Scripts\\python.exe toggle_accel.py --status
    .\\.venv\\Scripts\\python.exe toggle_accel.py --off     # 关掉（回滚）
    .\\.venv\\Scripts\\python.exe toggle_accel.py --on      # 打开（恢复 P3a）

改完要重启 WebUI / api_v2.py 才生效（模块在 import 时加载）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "GPT_SoVITS" / "Accel" / "PyTorch" / "AR" / "backends" / "sdpa_varlen_xpu.py"
DISABLED = BACKEND.with_suffix(".py.disabled")


def status() -> int:
    print("=" * 66)
    print("P3a 后端状态")
    print("=" * 66)
    print(f"  期望位置 : {BACKEND.relative_to(ROOT)}")
    print(f"  已启用   : {'是' if BACKEND.exists() else '否'}")
    print(f"  已禁用   : {'是' if DISABLED.exists() else '否'}")

    if BACKEND.exists():
        print("\n  当前推理会走 Accel（sdpa_varlen_xpu）")
        print("  优点：gpt_streaming 可用")
        print("  风险：驱动层崩溃时会 abort 整个进程")
    elif DISABLED.exists():
        print("\n  当前推理走原版 infer_panel（无 Accel）")
        print("  优点：稳，退回 P3a 之前的行为")
        print("  缺点：gpt_streaming 不可用")
    else:
        print("\n  [!] 两个文件都不在 —— P3a 可能根本没打上，"
              "或者被别的操作删了")
        print("      可以重跑 apply_p3a.py 恢复")
    print("=" * 66)
    return 0


def off() -> int:
    if DISABLED.exists() and not BACKEND.exists():
        print("已经是关闭状态，无需操作")
        return 0
    if not BACKEND.exists():
        print(f"[X] 找不到 {BACKEND.relative_to(ROOT)}")
        print("    是不是没打过 P3a？先跑 apply_p3a.py")
        return 1
    BACKEND.rename(DISABLED)
    print(f"[OK] 已禁用 -> {DISABLED.name}")
    print("     推理将退回原版 infer_panel 路径")
    print("     记得重启 WebUI / api_v2.py")
    return 0


def on() -> int:
    if BACKEND.exists() and not DISABLED.exists():
        print("已经是启用状态，无需操作")
        return 0
    if not DISABLED.exists():
        print(f"[X] 找不到 {DISABLED.relative_to(ROOT)}")
        print("    如果是全新环境，直接跑 apply_p3a.py")
        return 1
    DISABLED.rename(BACKEND)
    print(f"[OK] 已启用 -> {BACKEND.name}")
    print("     gpt_streaming 恢复可用，但之前那个崩溃可能复现")
    print("     记得重启 WebUI / api_v2.py")
    return 0


def verify(which: str) -> None:
    """跑一小段检查，确认改动后 resolve_acceleration 的返回值。"""
    print()
    print("验证（新进程）：")
    code = (
        "import sys, pathlib;"
        "sys.path.insert(0, str(pathlib.Path('.').resolve()));"
        "import torch;"
        "from tools.acceleration import resolve_acceleration, create_acceleration;"
        "print('  resolve_acceleration ->', resolve_acceleration('xpu', torch.float32))"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    out = (r.stdout or "").strip()
    if out:
        print(out)
    if r.returncode != 0:
        print("  (验证脚本没跑完，可能环境有问题)")
        print("  " + (r.stderr or "").strip().splitlines()[-1][:160])


def main() -> int:
    ap = argparse.ArgumentParser(description="开关 P3a 的 XPU 推理后端")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true", help="只看状态")
    g.add_argument("--off", action="store_true", help="禁用（回滚到原版路径）")
    g.add_argument("--on", action="store_true", help="启用（恢复 P3a）")
    args = ap.parse_args()

    if args.off:
        rc = off()
        if rc == 0:
            verify("off")
        return rc
    if args.on:
        rc = on()
        if rc == 0:
            verify("on")
        return rc
    return status()


if __name__ == "__main__":
    sys.exit(main())
