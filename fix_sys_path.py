#!/usr/bin/env python3
"""修 sys.path：让 tools 包在任何启动方式下都能被 import

背景
----
仓库里有一堆脚本用 `python <子目录>/xxx.py` 启动，此时 Python 把
脚本所在目录设为 sys.path[0]，而 `tools/` 在仓库根 —— 于是
`from tools... import ...` 直接 ModuleNotFoundError。

典型受害者：
    GPT_SoVITS/inference_webui.py:9      from tools.portable_runtime import ...
    GPT_SoVITS/inference_webui_fast.py:9 同上
    GPT_SoVITS/inference_cli.py:5       from tools.i18n.i18n import ...
    GPT_SoVITS/process_ckpt.py:7        同上
    tools/uvr5/webui.py:7               from tools.i18n.i18n import ...
    tools/asr/fasterwhisper_asr.py:12   from tools.asr.config import ...
    ...

注意 inference_webui.py 的坑：第 9 行 import tools，第 11 行才
activate_portable_runtime(change_cwd=True) 插路径 —— 鸡生蛋。

解法
----
往当前 venv 的 site-packages 写一个 .pth 文件。Python 启动时会把
每行内容当作路径加进 sys.path，一行搞定全部脚本，不动任何源码。

用法
----
    cd D:\\GPSVT\\GPT-SoVITS
    .\\.venv\\Scripts\\python.exe fix_sys_path.py            # 写入
    .\\.venv\\Scripts\\python.exe fix_sys_path.py --check    # 只检查
    .\\.venv\\Scripts\\python.exe fix_sys_path.py --remove   # 移除

副作用
------
* 只影响当前 venv（.pth 在 .venv/Lib/site-packages 里）
* 重建 venv 会丢，重跑一次即可
* 会把仓库根永久加进这个 venv 的 sys.path —— 对这个项目是好事，
  但如果这个 venv 还跑别的项目，可能造成同名包冲突（一般不会）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

PTH_NAME = "gsv_root.pth"


def site_packages() -> Path:
    """当前解释器的 site-packages 路径。"""
    # purelib 对 venv 就是 <venv>/Lib/site-packages
    return Path(sysconfig.get_paths()["purelib"])


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def pth_path() -> Path:
    return site_packages() / PTH_NAME


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def show_status() -> bool:
    pth = pth_path()
    root = repo_root()
    print(f"  解释器       : {sys.executable}")
    print(f"  在 venv 里吗 : {'是' if in_venv() else '否（注意：会写到全局 Python）'}")
    print(f"  site-packages: {site_packages()}")
    print(f"  仓库根       : {root}")
    print(f"  .pth 文件    : {pth}")

    exists = pth.exists()
    print(f"  .pth 存在    : {exists}")
    if exists:
        content = pth.read_text(encoding="utf-8", errors="replace").strip()
        print(f"  .pth 内容    : {content!r}")
        ok = content == str(root)
        print(f"  内容匹配     : {ok}")
        return ok
    return False


def write_pth() -> int:
    root = repo_root()
    toolspkg = root / "tools" / "__init__.py"
    if not toolspkg.exists():
        print(f"[ABORT] {toolspkg} 不存在，确认脚本放在仓库根目录")
        return 1

    pth = pth_path()
    pth.parent.mkdir(parents=True, exist_ok=True)
    pth.write_text(str(root) + "\n", encoding="utf-8")
    print(f"[OK] 已写入 {pth}")
    print(f"     内容: {root}")
    return 0


def remove_pth() -> int:
    pth = pth_path()
    if not pth.exists():
        print("[SKIP] .pth 不存在")
        return 0
    pth.unlink()
    print(f"[OK] 已删除 {pth}")
    return 0


def verify() -> int:
    """用子进程验证：从不同目录启动，tools 是否可导入。"""
    print()
    print("验证（每条都在新进程里跑）：")

    cases = [
        ("仓库根", repo_root(), ["-c", "import tools; print(tools.__file__)"]),
        ("GPT_SoVITS", repo_root() / "GPT_SoVITS",
         ["-c", "import tools; print(tools.__file__)"]),
        ("tools/uvr5", repo_root() / "tools" / "uvr5",
         ["-c", "import tools; print(tools.__file__)"]),
    ]

    all_ok = True
    for name, cwd, args in cases:
        if not cwd.exists():
            print(f"  [SKIP] {name}: 目录不存在")
            continue
        try:
            r = subprocess.run(
                [sys.executable] + args,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=str(cwd), timeout=120,
            )
            if r.returncode == 0:
                out = (r.stdout or "").strip().splitlines()
                print(f"  [OK]   {name:<12} -> {out[-1] if out else ''}")
            else:
                all_ok = False
                err = (r.stderr or "").strip().splitlines()
                print(f"  [FAIL] {name:<12} -> {err[-1] if err else 'unknown'}")
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {name:<12} -> {type(e).__name__}: {e}")

    # 真实入口的 import 检查（不启动 WebUI，只 import 到会失败的那一行之前）
    print()
    print("抽查真实入口的顶层 import：")
    checks = [
        ("GPT_SoVITS", "from tools.portable_runtime import activate"),
        ("GPT_SoVITS", "from tools.i18n.i18n import I18nAuto"),
        ("tools/uvr5", "from tools.i18n.i18n import I18nAuto"),
        ("tools/asr", "from tools.my_utils import load_cudnn"),
    ]
    for rel, stmt in checks:
        d = repo_root() / rel
        if not d.exists():
            continue
        try:
            r = subprocess.run(
                [sys.executable, "-c", stmt],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=str(d), timeout=120,
            )
            if r.returncode == 0:
                print(f"  [OK]   {rel:<12} {stmt[:52]}")
            else:
                all_ok = False
                err = (r.stderr or "").strip().splitlines()
                tail = err[-1] if err else "unknown"
                # Windows 上 psutil 之类缺依赖时不代表路径有问题
                print(f"  [INFO] {rel:<12} {stmt[:40]} -> {tail[:70]}")
        except Exception as e:
            print(f"  [INFO] {rel:<12} 跳过（{type(e).__name__}）")

    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="给 venv 写 .pth，让 tools 包全局可导入")
    ap.add_argument("--check", action="store_true", help="只检查现状，不修改")
    ap.add_argument("--remove", action="store_true", help="删除 .pth")
    ap.add_argument("--no-verify", action="store_true", help="写完不验证")
    args = ap.parse_args()

    print("=" * 70)
    print("sys.path 修复")
    print("=" * 70)
    print()

    if args.check:
        matched = show_status()
        return 0 if matched else 1

    if args.remove:
        return remove_pth()

    show_status()
    print()
    rc = write_pth()
    if rc != 0:
        return rc

    if not args.no_verify:
        rc = verify()

    print()
    print("=" * 70)
    if rc == 0:
        print("完成。现在可以直接跑：")
        print("    .\\.venv\\Scripts\\python.exe GPT_SoVITS\\inference_webui.py zh_CN")
    else:
        print("有检查项没通过，把上面输出贴出来。")
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main())
