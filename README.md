<div align="center">

<h1>GPT-SoVITS-For-Intel</h1>
适用于Intel显卡(XPU)的移植魔改版，绝大部分功能可以正常使用。<br><br>
</div>

## 快速开始

本项目为极简配置，没有打包成品，因此对于诸多资源需要手动拉取。
默认就是GPT-SoVITS源码+魔改部分，依照以下方式正确部署即可，

### venv配置

克隆本仓库
```bash
git clone https://github.com/LimeBlogs/GPT-SoVITS-For-Intel.git
```

进入本仓库

```bash
cd GPT-SoVITS-For-Intel
```

使用venv(推荐)
*Python 3.12+*

```bash
python -m venv .venv
```

创建指定版本（默认PATH不是所适用版本）
以3.12.10为例

```bash
py -3.12 -m venv .venv
```

### 激活环境

激活环境(Windows)

```bash
.\.venv\Scripts\activate
```

对于MacOS/Linux

```bash
source .venv/bin/activate
```

### 安装依赖

默认使用南京大学源，你要有流量我没意见。

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 `
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu
```

依旧

```bash
pip install -r requirements-xpu-win.txt
```

#### 下载必要模型
该脚本已配置好镜像源，可以直接使用。

```bash 
python download_models.py
```

### 启动

修复系统路径（一次即可）

```bash
python fix_sys_path.py
```

```bash
python webui.py
```

## 不知道