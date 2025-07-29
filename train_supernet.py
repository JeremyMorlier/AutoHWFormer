import sys
import os
import time
import datetime
import yaml
import json
import math
import random
from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup

# from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy, ModelEma
from timm.utils.model import unwrap_model

import utils
from args import get_autoformer_argsparse
from logger import Logger

from libAutoFormer.config import cfg, update_config_from_file
from libAutoFormer.samplers import RASampler
from datasets.autoformer_datasets import build_dataset
from model.supernet_transformer import Vision_TransformerSuper
from references import MetricLogger, SmoothedValue


def sample_configs(choices):
    config = {}
    dimensions = ["mlp_ratio", "num_heads"]
    depth = random.choice(choices["depth"])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config["embed_dim"] = [random.choice(choices["embed_dim"])] * depth

    config["layer_num"] = depth
    return config


def select_config(model: torch.nn.Module, config, mode="super", retrain_config=None):
    model_module = unwrap_model(model)
    if mode == "retrain":
        config = retrain_config
    model_module.set_sample_config(config=config)
    return model_module, model_module.get_sampled_params_numel(config)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    amp: bool = True,
    teacher_model: torch.nn.Module = None,
    teach_loss: torch.nn.Module = None,
    config=None,
    choices=None,
    mode="super",
    retrain_config=None,
):
    model.train()
    criterion.train()

    print("torch cudnn dot product enabled:", torch.backends.cuda.cudnn_sdp_enabled())
    # set random seed
    random.seed(epoch)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    if mode == "retrain":
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    if config is not None:
        selected_config = config
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Sample Config
        if config is None:
            selected_config = sample_configs(choices)
        _, config_params = select_config(model, selected_config, mode, retrain_config)
        while config is None and config_params >= 3e7:
            selected_config = sample_configs(choices)
            _, config_params = select_config(model, selected_config, mode, retrain_config)
        # # sample random config
        # if mode == 'super':
        #     config = sample_configs(choices=choices)
        #     model_module = unwrap_model(model)
        #     model_module.set_sample_config(config=config)
        # elif mode == 'retrain':
        #     config = retrain_config
        #     model_module = unwrap_model(model)
        #     model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.autocast(device.type):
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    amp=True,
    config=None,
    choices=None,
    mode="super",
    retrain_config=None,
):
    print("torch cudnn dot product enabled:", torch.backends.cuda.cudnn_sdp_enabled())
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    if mode == "super":
        if config is not None:
            selected_config = config
        else:
            selected_config = sample_configs(choices)
        # Sample Config
        model_module, config_params = select_config(model, selected_config, mode, retrain_config)
        while config is None and config_params >= 3e7:
            selected_config = sample_configs(choices)
            model_module, config_params = select_config(model, selected_config, mode, retrain_config)
    elif mode == "retrain":
        selected_config = retrain_config
        model_module, _ = select_config(model, selected_config, mode, retrain_config)
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    # if mode == 'super':
    #     config = sample_configs(choices=choices)
    #     model_module = unwrap_model(model)
    #     model_module.set_sample_config(config=config)
    # else:
    #     config = retrain_config
    #     model_module = unwrap_model(model)
    #     model_module.set_sample_config(config=config)

    print("sampled model config: {}".format(selected_config))
    parameters = model_module.get_sampled_params_numel(selected_config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.autocast(device.type):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_dataloaders(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Get Samplers
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(args.batch_size),
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_train, data_loader_val


def main(args):
    utils.init_distributed_mode(args)
    utils.init_signal_handler()

    device = torch.device(args.device)
    update_config_from_file(args.cfg)
    if utils.is_main_process():
        # similar API to wandb except mode and log_dir
        logger = Logger(
            project_name="whatever",
            run_name=args.name,
            tags=["patate"],
            resume=True,
            args=args,
            mode=args.logger,
            log_dir=args.output_dir,
        )

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(args_text)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    data_loader_train, data_loader_val = get_dataloaders(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = Vision_TransformerSuper(
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
    )

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Parallelize the model using Distributed Data Parallel (DDP)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print(
        "cuda_mem_allocated",
        torch.cuda.memory_allocated(),
        "cuda_max_mem_allocated",
        torch.cuda.max_memory_allocated(),
        "cuda_mem_reserved",
        torch.cuda.memory_reserved(),
        "cuda_max_mem_reserved",
        torch.cuda.max_memory_reserved(),
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    if utils.is_main_process():
        logger.log(
            {
                "nparams": n_parameters,
                "cuda_mem_allocated": torch.cuda.memory_allocated(),
            }
        )

    # TODO: as in https://github.com/microsoft/Cream/blob/main/AutoFormer/supernet_train.py add teacher model for distillation
    teacher_model = None
    teacher_loss = None

    model_ema = None

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        elif os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        else:
            checkpoint = None

        if checkpoint is not None:
            print("Resuming from checkpoint '{}'".format(args.resume))
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
        else:
            print("No checkpoint found, starting from scratch")

    retrain_config = None
    if args.mode == "retrain" and "RETRAIN" in cfg:
        retrain_config = {
            "layer_num": cfg.RETRAIN.DEPTH,
            "embed_dim": [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
            "num_heads": cfg.RETRAIN.NUM_HEADS,
            "mlp_ratio": cfg.RETRAIN.MLP_RATIO,
        }
    if args.eval:
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            mode=args.mode,
            retrain_config=retrain_config,
        )
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            amp=args.amp,
            teacher_model=teacher_model,
            teach_loss=teacher_loss,
            choices=choices,
            mode=args.mode,
            retrain_config=retrain_config,
        )
        if utils.is_main_process():
            logger.log(train_stats)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            amp=args.amp,
            choices=choices,
            mode=args.mode,
            retrain_config=retrain_config,
        )
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        print(f"Epoch {epoch} training time: {time.time() - epoch_start_time:.2f}s")
        if utils.is_main_process():
            logger.log(test_stats)
            logger.log({"epoch_time": time.time() - epoch_start_time})

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if utils.is_main_process():
            logger.log(log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = get_autoformer_argsparse()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
