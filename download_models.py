#!/usr/bin/env python3
"""下载 GPT-SoVITS 预训练模型（ModelScope 源）

背景
----
仓库 clone 下来时 GPT_SoVITS/pretrained_models/ 是空的，只有 .gitignore。
缺了它推理会报：
    HFValidationError: Repo id must be in the form 'repo_name' or ...
因为 transformers 找不到本地目录，就把路径字符串当成 HF 仓库名去解析。

下载源（官方 install.sh 用的同一套）
------------------------------------
    https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/

体积（实测）
------------
    pretrained_models.zip   4.35 GB   ← 推理/训练必须
    G2PWModel.zip           561 MB    ← 中文多音字（G2PW=0 时可跳过）
    uvr5_weights.zip        499 MB    ← 人声分离
    nltk_data.zip           9.5 MB    ← 文本处理
    open_jtalk_dic         22.6 MB    ← 日文

用法
----
    cd D:\\GPSVT\\GPT-SoVITS
    .\\.venv\\Scripts\\python.exe download_models.py --check      # 只看现状
    .\\.venv\\Scripts\\python.exe download_models.py              # 只下 pretrained（4.35G）
    .\\.venv\\Scripts\\python.exe download_models.py --only all   # 全下（约 5.4G）
    .\\.venv\\Scripts\\python.exe download_models.py --only g2pw uvr5

断点续传：自动用 curl.exe（Win10+ 自带）的 -C - ，断了重跑同一条命令即可。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"

# name -> (zip 文件名, 解压目标目录(相对仓库根), 校验用的哨兵路径, 大约 MB)
ITEMS = {
    "pretrained": (
        "pretrained_models.zip", "GPT_SoVITS",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base", 4352,
    ),
    "g2pw": (
        "G2PWModel.zip", "GPT_SoVITS/text",
        "GPT_SoVITS/text/G2PWModel", 562,
    ),
    "uvr5": (
        "uvr5_weights.zip", "tools/uvr5",
        "tools/uvr5/uvr5_weights", 499,
    ),
    "nltk": (
        "nltk_data.zip", ".",
        "nltk_data", 10,
    ),
    "jtalk": (
        "open_jtalk_dic_utf_8-1.11.tar.gz", ".",
        "open_jtalk_dic_utf_8-1.11", 23,
    ),
}

ROOT = Path(__file__).resolve().parent
DL_DIR = ROOT / "_downloads"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def free_space_gb(p: Path) -> float:
    try:
        return shutil.disk_usage(p).free / (1024 ** 3)
    except Exception:
        return -1


def has_content(p: Path) -> bool:
    """目录只含 .gitignore 视为空（仓库里 pretrained_models/ 就是这样）。"""
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_size > 0
    for f in p.rglob("*"):
        if f.is_file() and f.name != ".gitignore":
            return True
    return False


def which_curl() -> str | None:
    for c in ("curl.exe", "curl"):
        p = shutil.which(c)
        if p:
            return p
    return None


# ------------------------------------------------------------------ 下载

def download_curl(url: str, dest: Path) -> bool:
    curl = which_curl()
    if curl is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [curl, "-L", "-C", "-", "--retry", "10", "--retry-delay", "3",
           "--connect-timeout", "30", "-o", str(dest), url]
    print(f"  $ {' '.join(cmd[:6])} ... {dest.name}")
    # curl 自带进度条，直接让它输出到控制台
    r = subprocess.run(cmd)
    return r.returncode == 0


def download_urllib(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header("Range", f"bytes={pos}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0) + pos
            mode = "ab" if pos and resp.status == 206 else "wb"
            if mode == "wb":
                pos = 0
            with open(dest, mode) as f:
                last = time.time()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
                    if time.time() - last > 0.5:
                        last = time.time()
                        pct = f"{pos / total * 100:5.1f}%" if total else "  ?  "
                        print(f"\r    {pct}  {human(pos)}"
                              + (f" / {human(total)}" if total else ""), end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  [FAIL] {type(e).__name__}: {e}")
        return False


def download(url: str, dest: Path) -> bool:
    if download_curl(url, dest):
        return True
    print("  (curl 不可用，退回 urllib)")
    return download_urllib(url, dest)


# ------------------------------------------------------------------ 解压

def extract_zip(zip_path: Path, target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.namelist()
            total = len(members)
            print(f"  解压 {total} 个条目 -> {target_dir}")
            t0 = time.time()
            for i, name in enumerate(members, 1):
                z.extract(name, target_dir)
                if i % 200 == 0 or i == total:
                    print(f"\r    {i}/{total}  ({i / total * 100:5.1f}%)  "
                          f"{time.time() - t0:.0f}s", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n  [FAIL] 解压失败 {type(e).__name__}: {e}")
        return False


def extract_tar(tar_path: Path, target_dir: Path) -> bool:
    import tarfile
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as t:
            t.extractall(target_dir)
        print(f"  解压完成 -> {target_dir}")
        return True
    except Exception as e:
        print(f"  [FAIL] 解压失败 {type(e).__name__}: {e}")
        return False


# ------------------------------------------------------------------ 主流程

def check_only() -> int:
    print()
    print(f"{'项':<12} {'状态':<8} {'大小':>10}  说明")
    print("-" * 74)
    missing = []
    for key, (fname, tdir, sentinel, mb) in ITEMS.items():
        sp = ROOT / sentinel
        if has_content(sp):
            size = sum(p.stat().st_size for p in sp.rglob("*") if p.is_file()) \
                if sp.is_dir() else sp.stat().st_size
            print(f"{key:<12} {'OK':<8} {human(size):>10}  {sentinel}")
        else:
            print(f"{key:<12} {'缺失':<8} {'~%d MB' % mb:>10}  {sentinel}")
            missing.append(key)
    print()
    if missing:
        print(f"缺 {len(missing)} 项：{', '.join(missing)}")
        need = sum(ITEMS[k][3] for k in missing) / 1024
        print(f"预计需要下载约 {need:.1f} GB，解压后再占约 {need:.1f} GB")
        print(f"目标盘剩余：{free_space_gb(ROOT):.1f} GB")
    else:
        print("全部就位。")
    return 0 if not missing else 1


def do_item(key: str, keep_zip: bool) -> bool:
    fname, tdir, sentinel, mb = ITEMS[key]
    sp = ROOT / sentinel
    target = ROOT / tdir

    print()
    print("=" * 70)
    print(f"[{key}] {fname}  (~{mb} MB) -> {target}")
    print("=" * 70)

    if has_content(sp):
        print(f"  已存在，跳过：{sentinel}")
        return True

    free = free_space_gb(ROOT)
    if 0 <= free < mb / 1024 * 2.2:
        print(f"  [ABORT] 磁盘剩余 {free:.1f} GB，下载 + 解压需要约 {mb / 1024 * 2.2:.1f} GB")
        return False

    DL_DIR.mkdir(parents=True, exist_ok=True)
    dest = DL_DIR / fname
    url = f"{BASE}/{fname}"

    if dest.exists():
        print(f"  发现已下载一半的文件：{human(dest.stat().st_size)}，尝试续传")

    t0 = time.time()
    if not download(url, dest):
        print("  下载失败。可重跑同一条命令续传。")
        return False
    print(f"  下载完成 {human(dest.stat().st_size)}，用时 {time.time() - t0:.0f}s")

    t1 = time.time()
    if fname.endswith(".zip"):
        ok = extract_zip(dest, target)
    elif fname.endswith(".tar.gz"):
        ok = extract_tar(dest, target)
    else:
        ok = False
        print("  未知压缩格式")
    if not ok:
        return False
    print(f"  解压用时 {time.time() - t1:.0f}s")

    if keep_zip:
        print(f"  保留压缩包：{dest}")
    else:
        try:
            dest.unlink()
            print("  已删除压缩包")
        except Exception:
            pass

    if not has_content(sp):
        print(f"  [WARN] 解压后仍找不到 {sentinel}（或只有 .gitignore）")
        print(f"         目标目录内容：{list(target.iterdir())[:8] if target.exists() else '(不存在)'}")
        return False

    print(f"  [OK] {sp}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 GPT-SoVITS 预训练模型")
    ap.add_argument("--check", action="store_true", help="只检查现状")
    ap.add_argument("--only", nargs="+", default=["pretrained"],
                    choices=list(ITEMS.keys()) + ["all"],
                    help="要下哪些，默认 pretrained")
    ap.add_argument("--keep-zip", action="store_true", help="解压后保留压缩包")
    args = ap.parse_args()

    print("=" * 70)
    print("GPT-SoVITS 预训练模型下载")
    print("=" * 70)
    print(f"仓库：{ROOT}")
    print(f"目标盘剩余：{free_space_gb(ROOT):.1f} GB")

    if not (ROOT / "GPT_SoVITS").is_dir():
        print("[ABORT] 这里不像 GPT-SoVITS 仓库根，脚本要放在仓库根目录")
        return 1

    if args.check:
        return check_only()

    keys = list(ITEMS.keys()) if "all" in args.only else args.only

    results = {}
    for k in keys:
        try:
            results[k] = do_item(k, args.keep_zip)
        except KeyboardInterrupt:
            print("\n中断。已下载的部分保留在 _downloads/，重跑可续传。")
            return 130

    print()
    print("=" * 70)
    ok = all(results.values())
    for k, v in results.items():
        print(f"  {k:<12} {'OK' if v else 'FAIL'}")
    print("=" * 70)

    if ok:
        print()
        print("验证一下：")
        print("  .\\.venv\\Scripts\\python.exe download_models.py --check")
        print("  .\\.venv\\Scripts\\python.exe GPT_SoVITS\\inference_webui.py zh_CN")
    else:
        print()
        print("有失败项。断了直接重跑同一条命令，curl -C - 会续传。")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
