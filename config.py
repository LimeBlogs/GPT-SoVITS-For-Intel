import os
import re
import sys

import torch

from tools import device_utils as du

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language=os.environ.get("language", "Auto"))


pretrained_sovits_name = {
    "v1": "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",  ###v3v4还要检查vocoder，算了。。。
    "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "v5dev": "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "v5turbo": "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}

pretrained_gpt_name = {
    "v1": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "v3": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v4": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v5dev": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v5turbo": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2Pro": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}
name2sovits_path = {
    i18n("不训练直接推v5dev底模！"): pretrained_sovits_name["v5dev"],
    i18n("不训练直接推v5turbo底模！"): pretrained_sovits_name["v5turbo"],
    # i18n("不训练直接推v1底模！"): "GPT_SoVITS/pretrained_models/s2G488k.pth",
    i18n("不训练直接推v2底模！"): "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    # i18n("不训练直接推v3底模！"): "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    # i18n("不训练直接推v4底模！"): "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    i18n("不训练直接推v2Pro底模！"): "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    i18n("不训练直接推v2ProPlus底模！"): "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}
name2gpt_path = {
    # i18n("不训练直接推v1底模！"):"GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    i18n(
        "不训练直接推v2底模！"
    ): "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    i18n("不训练直接推v3底模！"): "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}
SoVITS_weight_root = [
    "SoVITS_weights",
    "SoVITS_weights_v2",
    "SoVITS_weights_v3",
    "SoVITS_weights_v4",
    "SoVITS_weights_v5dev",
    "SoVITS_weights_v5turbo",
    "SoVITS_weights_v2Pro",
    "SoVITS_weights_v2ProPlus",
]
GPT_weight_root = [
    "GPT_weights",
    "GPT_weights_v2",
    "GPT_weights_v3",
    "GPT_weights_v4",
    "GPT_weights_v5dev",
    "GPT_weights_v5turbo",
    "GPT_weights_v2Pro",
    "GPT_weights_v2ProPlus",
]
SoVITS_weight_version2root = {
    "v1": "SoVITS_weights",
    "v2": "SoVITS_weights_v2",
    "v3": "SoVITS_weights_v3",
    "v4": "SoVITS_weights_v4",
    "v5dev": "SoVITS_weights_v5dev",
    "v5turbo": "SoVITS_weights_v5turbo",
    "v2Pro": "SoVITS_weights_v2Pro",
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
}
GPT_weight_version2root = {
    "v1": "GPT_weights",
    "v2": "GPT_weights_v2",
    "v3": "GPT_weights_v3",
    "v4": "GPT_weights_v4",
    "v5dev": "GPT_weights_v5dev",
    "v5turbo": "GPT_weights_v5turbo",
    "v2Pro": "GPT_weights_v2Pro",
    "v2ProPlus": "GPT_weights_v2ProPlus",
}


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def get_weights_names():
    SoVITS_names = []
    for key in name2sovits_path:
        if os.path.exists(name2sovits_path[key]):
            SoVITS_names.append(key)
    for path in SoVITS_weight_root:
        if not os.path.exists(path):
            continue
        for name in os.listdir(path):
            if name.endswith(".pth"):
                SoVITS_names.append("%s/%s" % (path, name))
    if not SoVITS_names:
        SoVITS_names = [""]
    GPT_names = []
    for key in name2gpt_path:
        if os.path.exists(name2gpt_path[key]):
            GPT_names.append(key)
    for path in GPT_weight_root:
        if not os.path.exists(path):
            continue
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                GPT_names.append("%s/%s" % (path, name))
    SoVITS_names = sorted(SoVITS_names, key=custom_sort_key)
    GPT_names = sorted(GPT_names, key=custom_sort_key)
    if not GPT_names:
        GPT_names = [""]
    return SoVITS_names, GPT_names


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": SoVITS_names, "__type__": "update"}, {
        "choices": GPT_names,
        "__type__": "update",
    }


# 推理用的指定模型
sovits_path = ""
gpt_path = ""
is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == "true" else False
is_share_str = os.environ.get("is_share", "False")
is_share = True if is_share_str.lower() == "true" else False

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

exp_root = "logs"
python_exec = sys.executable or "python"

webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871

api_port = 9880


# Thanks to the contribution of @Karasukaigan and @XXXXRT666
def get_device_dtype_sm(idx: int, dev_type: str = "cuda") -> tuple[torch.device, torch.dtype, float, float]:
    """返回 (device, dtype, 排序键1, 排序键2)，保持原签名兼容。

    本机（A750）预期落到 fp32：
      has_bf16=False, has_xmx=False → 后面两个分支都不进。
    """
    cpu = torch.device("cpu")
    if not du.accel_available(dev_type):
        return cpu, torch.float32, 0.0, 0.0

    dev = torch.device(f"{dev_type}:{idx}")
    p = du.properties(idx, dev_type)
    mem_gb = p["total_memory_gb"]

    if dev_type == "cuda":
        rank_key = p["arch"][0] + p["arch"][1] / 10.0
        is_16_series = bool(re.search(r"16\d{2}", p["name"])) and rank_key == 7.5
    else:
        # Intel：Xe 核心数当排序键（A750 = 448EU → 4.48）
        rank_key = p["compute_units"] / 100.0
        is_16_series = False

    # 独显：total_memory 就是真实 VRAM
    min_mem = 6.0 if (dev_type == "xpu" and p.get("is_integrated")) else 4.0
    if mem_gb < min_mem:
        return cpu, torch.float32, 0.0, 0.0

    if dev_type == "xpu":
        if p["has_bf16"]:                      # 本机 False
            return dev, torch.bfloat16, rank_key, mem_gb
        if du.supports_fp16_fast(dev):         # 本机 False
            return dev, torch.float16, rank_key, mem_gb
        return dev, torch.float32, rank_key, mem_gb      # ← 本机走这里

    if dev_type == "cuda":
        if rank_key == 6.1 or is_16_series:
            return dev, torch.float32, rank_key, mem_gb
        if rank_key > 6.1:
            return dev, torch.float16, rank_key, mem_gb

    return cpu, torch.float32, 0.0, 0.0


IS_GPU = True
GPU_INFOS: list[str] = []
GPU_INDEX: set[int] = set()
_BACKEND = du.backend_type()
GPU_COUNT = du.device_count(_BACKEND)
CPU_INFO: str = "0\tCPU " + i18n("CPU训练,较慢")
tmp: list[tuple[torch.device, torch.dtype, float, float]] = []
memset: set[float] = set()

for i in range(max(GPU_COUNT, 1)):
    tmp.append(get_device_dtype_sm(i, _BACKEND))

for j in tmp:
    memset.add(j[3])
    if j[0].type != "cpu":
        GPU_INFOS.append(f"{j[0].index}\t{du.properties(j[0].index, _BACKEND)['name']}")
        GPU_INDEX.add(j[0].index)

if not GPU_INFOS:
    IS_GPU = False
    GPU_INFOS.append(CPU_INFO)
    GPU_INDEX.add(0)

infer_device = max(tmp, key=lambda x: (x[2], x[3]))[0]
is_half = any(dt in (torch.float16, torch.bfloat16) for _, dt, _, _ in tmp)


class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device

        self.webui_port_main = webui_port_main
        self.webui_port_uvr5 = webui_port_uvr5
        self.webui_port_infer_tts = webui_port_infer_tts
        self.webui_port_subfix = webui_port_subfix

        self.api_port = api_port
