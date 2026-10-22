# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/data_module.py
# reference: https://github.com/lifeiteng/vall-e
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from AR.data.bucket_sampler import DistributedBucketSampler
from AR.data.dataset import Text2SemanticDataset

import os  # noqa: E402


class Text2SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        train_semantic_path,
        train_phoneme_path,
        dev_semantic_path=None,
        dev_phoneme_path=None,
    ):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.num_workers = self.config["data"]["num_workers"]

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path,
            max_sec=self.config["data"]["max_sec"],
            pad_val=self.config["data"]["pad_val"],
        )
        self._dev_dataset = self._train_dataset
        # self._dev_dataset = Text2SemanticDataset(
        #     phoneme_path=self.dev_phoneme_path,
        #     semantic_path=self.dev_semantic_path,
        #     max_sample=self.config['data']['max_eval_sample'],
        #     max_sec=self.config['data']['max_sec'],
        #     pad_val=self.config['data']['pad_val'])

    def train_dataloader(self):
        batch_size = (
            self.config["train"]["batch_size"] // 2
            if self.config["train"].get("if_dpo", False) is True
            else self.config["train"]["batch_size"]
        )
        batch_size = max(min(batch_size, len(self._train_dataset) // 4), 1)  # 防止不保存
        sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size)
        # R3：Windows 上 DataLoader worker 用 spawn 启动，会重新 import 主模块 → 必须 0
        num_workers = 0 if os.name == "nt" else self.num_workers
        kwargs = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 16
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=num_workers,
            **kwargs,
        )

    def val_dataloader(self):
        # 验证集用 1（0 在某些 Lightning 版本会警告），其余平台沿用 max(.,12)
        num_workers = max(self.num_workers, 1) if os.name == "nt" else max(self.num_workers, 12)
        kwargs = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 16
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
            num_workers=num_workers,
            **kwargs,
        )

    # 这个会使用到嘛？
    def test_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
        )
