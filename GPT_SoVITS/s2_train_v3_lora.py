import warnings

warnings.filterwarnings("ignore")

import os
import pathlib
import sys

# ── R1：路径垫片（GPT_SoVITS/*.py → parents[1]）──
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import amp_utils as amp  # noqa: E402
from tools import device_utils as du  # noqa: E402

import utils  # noqa: E402

hps = utils.get_hparams(stage=2)

# ── 设备可见性（原写法写死 CUDA_VISIBLE_DEVICES）──
_BACKEND = du.backend_type()
_GPU_NUMS = str(getattr(hps.train, "gpu_numbers", "0")).replace("-", ",")  # R2：HParams 没有 .get()
if _BACKEND == "xpu":
    os.environ["ZE_AFFINITY_MASK"] = _GPU_NUMS
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", f"level_zero:{_GPU_NUMS.split(',')[0]}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_NUMS

# 必须在 module.data_utils 被 import 之前设置（模块级读取）
os.environ["version"] = hps.model.version

import logging  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from collections import OrderedDict as od  # noqa: E402
from random import randint  # noqa: E402
from time import time as ttime  # noqa: E402

from module import commons  # noqa: E402
from module.data_utils import (  # noqa: E402
    DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerLoaderV4,
)
from module.models import (  # noqa: E402
    SynthesizerTrnV3 as SynthesizerTrn,
)
from peft import LoraConfig, get_peft_model  # noqa: E402
from process_ckpt import savee  # noqa: E402

# ── 后端特性开关 ──
if _BACKEND == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0

device = str(du.default_device())
# 本机：supports_bf16() 因 has_bfloat16_conversions=False 返回 False → "fp32"
_PRECISION = "bf16" if (torch.device(device).type == "xpu" and du.supports_bf16(device)) else "fp32"
_TORCH_DTYPE = torch.bfloat16 if _PRECISION == "bf16" else torch.float32


def _dev(rank: int = 0) -> torch.device:
    if du.accel_available(_BACKEND):
        return torch.device(f"{_BACKEND}:{rank}")
    return torch.device("cpu")


def _pg_backend() -> str:
    if _BACKEND == "xpu":
        return "ccl"  # oneCCL，仅多卡；单卡不会走到
    return "gloo" if os.name == "nt" else "nccl"


def main():
    n_gpus = du.device_count(_BACKEND) if du.accel_available(_BACKEND) else 1
    want_ddp = bool(getattr(hps.train, "ddp", False))  # R2：HParams 没有 .get()
    if n_gpus <= 1 or not want_ddp:
        run(0, 1, hps)
        return

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    use_ddp = du.accel_available(_BACKEND) and n_gpus > 1
    try:
        if use_ddp:
            dist.init_process_group(
                backend=_pg_backend(),
                init_method="env://?use_libuv=False",
                world_size=n_gpus,
                rank=rank,
            )
        _run(rank, n_gpus, hps, use_ddp)
    finally:
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()


def _run(rank, n_gpus, hps, use_ddp):
    global global_step, no_grad_names, save_root, lora_rank
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    du.manual_seed_all(hps.train.seed)
    if du.accel_available(_BACKEND):
        du.set_device(rank, _BACKEND)

    TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3 if hps.model.version == "v3" else TextAudioSpeakerLoaderV4
    TextAudioSpeakerCollate = TextAudioSpeakerCollateV3 if hps.model.version == "v3" else TextAudioSpeakerCollateV4
    train_dataset = TextAudioSpeakerLoader(hps.data)  ########
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            # 1100,
            # 1200,
            # 1300,
            # 1400,
            # 1500,
            # 1600,
            # 1700,
            # 1800,
            # 1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    worker_count = 0 if os.name == "nt" else 4  # R3：Windows 的 spawn worker 会重新 import 本模块
    loader_kwargs = dict(
        num_workers=worker_count,
        shuffle=False,
        pin_memory=du.accel_available(_BACKEND),  # F9：原写法 torch.cuda.is_available()
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )
    save_root = "%s/logs_s2_%s_lora_%s" % (hps.data.exp_dir, hps.model.version, hps.train.lora_rank)
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=True,
    )

    def get_model(hps):
        return SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

    def get_optim(net_g):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),  ###默认所有层lr一致
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    def model2dev(net_g, rank):
        # 顺序不能错：先搬设备，再注入 LoRA（1f）
        net_g = net_g.to(_dev(rank))
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        # fp32 下不需要额外的 .to(dtype)；保留分支以便将来启用 bf16
        if _PRECISION == "bf16":
            net_g = net_g.to(_TORCH_DTYPE)
        if use_ddp:
            net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        return net_g

    try:  # 如果能加载自动resume
        net_g = get_model(hps)
        net_g = model2dev(net_g, rank)
        optim_g = get_optim(net_g)
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        net_g = get_model(hps)
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G != None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2G,
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)["weight"],
                    strict=False,
                ),
            )
        net_g = model2dev(net_g, rank)
        optim_g = get_optim(net_g)

    no_grad_names = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name.replace("module.", ""))
            # print(name, "not requires_grad")
    # print(no_grad_names)
    # os._exit(233333)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str):
        scheduler_g.step()

    scaler = amp.make_scaler(device, hps.train.fp16_run and _PRECISION != "bf16")

    net_d = optim_d = scheduler_d = None
    print("start training from epoch %s" % epoch_str)
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                # [train_loader, eval_loader], logger, [writer, writer_eval])
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
    print("training done")


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    t_last = ttime()
    for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(
        tqdm(train_loader)
    ):
        _t = _dev(rank)
        spec = spec.to(_t, non_blocking=True)
        spec_lengths = spec_lengths.to(_t, non_blocking=True)
        mel = mel.to(_t, non_blocking=True)
        mel_lengths = mel_lengths.to(_t, non_blocking=True)
        ssl = ssl.to(_t, non_blocking=True)
        ssl.requires_grad = False
        text = text.to(_t, non_blocking=True)
        text_lengths = text_lengths.to(_t, non_blocking=True)

        with amp.autocast_ctx(device, hps.train.fp16_run and _PRECISION != "bf16", _TORCH_DTYPE):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [cfm_loss]
                logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

        global_step += 1
        if rank == 0 and global_step % 20 == 0:
            # 每 20 step 同步一次，取平均单步耗时 + 显存
            du.synchronize(device)
            now = ttime()
            ms_per_step = (now - t_last) * 1000.0 / 20.0
            t_last = now
            print(f"[perf] step {global_step}  {ms_per_step:.1f} ms/step  {du.memory_report(device, 's2')}")
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(233333333333)),
            )
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            sim_ckpt = od()
            for key in ckpt:
                # if "cfm"not in key:
                #     print(key)
                if key not in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        sim_ckpt,
                        hps.name + "_e%s_s%s_l%s" % (epoch, global_step, lora_rank),
                        epoch,
                        global_step,
                        hps,
                        model_version=hps.model.version,
                        lora_rank=lora_rank,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


if __name__ == "__main__":
    main()
