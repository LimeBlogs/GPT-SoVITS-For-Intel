# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import device_utils as du  # noqa: E402

import os

_BACKEND = du.backend_type()

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    _ids = os.environ["_CUDA_VISIBLE_DEVICES"]
    if _BACKEND == "xpu":
        os.environ["ZE_AFFINITY_MASK"] = _ids
        os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", f"level_zero:{_ids.split(',')[0]}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ids
import argparse
import logging
import platform
from pathlib import Path

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt
from process_ckpt import my_save


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    # torch.save(
                    # print(os.environ)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    # 音素符号表（symbols / symbols2）由 AR/data/dataset.py 在运行时读 os.environ["version"] 决定
    os.environ["version"] = str(config.get("version") or os.environ.get("version") or "v2")
    os.environ.setdefault("hz", "25hz")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"

    _HAS_ACCEL = du.accel_available(_BACKEND)
    # config 来自 yaml.full_load → 普通 dict，.get() 可用（与 hps 不同）
    _MULTI = _HAS_ACCEL and du.device_count(_BACKEND) > 1 and bool(config["train"].get("ddp", False))
    # 本机走 fp32：XPU 恒用 32-true（没启用 XMX，bf16 无收益）
    _precision = "32-true" if _BACKEND == "xpu" else config["train"]["precision"]

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if _HAS_ACCEL else "cpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=(du.device_count(_BACKEND) if _MULTI else 1) if _HAS_ACCEL else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=(
            DDPStrategy(
                process_group_backend=(
                    "ccl" if _BACKEND == "xpu" else ("gloo" if platform.system() == "Windows" else "nccl")
                )
            )
            if _MULTI
            else "auto"
        ),
        precision=_precision,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    # args for dataset
    # parser.add_argument('--train_semantic_path',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/6-name2semantic.tsv')
    # parser.add_argument('--train_phoneme_path', type=str, default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/2-name2text.txt')

    # parser.add_argument('--dev_semantic_path', type=str, default='dump_mix/semantic_dev.tsv')
    # parser.add_argument('--dev_phoneme_path', type=str, default='dump_mix/phoneme_dev.npy')
    # parser.add_argument('--output_dir',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/logs_s1',help='directory to save the results')
    # parser.add_argument('--output_dir',type=str,default='/liujing04/gpt_logs/s1/xuangou_ft',help='directory to save the results')

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
