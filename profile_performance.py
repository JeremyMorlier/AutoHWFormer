import torch

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from timm.utils.model import unwrap_model

import utils
from args import get_profile_argparse

from libAutoFormer.config import cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper

from torch.profiler import profile, record_function, ProfilerActivity
from train_supernet import sample_configs, select_config


def profile_training(
    model,
    inputs,
    targets,
    criterion,
    optimizer,
    activities,
    choices,
    torch_profile=True,
):
    model.train()
    if torch_profile:
        with profile(activities=activities) as prof:
            for i in range(0, 5):
                with record_function("selecting subnet"):
                    config = sample_configs(choices=choices)
                    model_module = unwrap_model(model)
                    model_module.set_sample_config(config=config)
                with record_function("model_train"):
                    output = model(inputs)
                    loss = criterion(output, targets)
                with record_function("backward+OPT"):
                    loss.backward()
                    optimizer.step()
        return prof
    else:
        for i in range(0, 100):
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def profile_evaluate(model, inputs, activities, torch_profile=True):
    model.eval()
    if torch_profile:
        with profile(
            activities=activities, profile_memory=True, record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                _ = model(inputs)

        return prof
    else:
        for i in range(0, 100):
            model(inputs)


def main(args):
    utils.init_distributed_mode(args)
    utils.init_signal_handler()

    device = torch.device(args.device)
    update_config_from_file(args.cfg)

    # As no dataset is used, manually update args
    args.nb_classes = 1000

    # Torch profile measured devices
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sort_by_keyword = args.device + "_time_total"

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

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
        profile=True,
    )
    model.to(device)
    # model.compile(mode="reduce-overhead", fullgraph=True)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Parallelize the model using Distributed Data Parallel (DDP)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    _ = NativeScaler()
    _, _ = create_scheduler(args, optimizer)

    # criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    config = sample_configs(choices=choices)
    select_config(model, config)
    random_inputs = torch.rand(100, 3, 224, 224).to(device)
    random_targets = torch.rand(100, 1000).to(device)

    print(model_without_ddp.blocks[0].attn.get_complexity(196 + 1))
    profile_train = profile_training(
        model,
        random_inputs,
        random_targets,
        criterion=criterion,
        optimizer=optimizer,
        activities=activities,
        choices=choices,
        torch_profile=True,
    )

    profile_train.export_chrome_trace(f"trace_train_{args.gpu}.json")
    profile_eval = profile_evaluate(
        model, random_inputs, activities=activities, torch_profile=True
    )

    profile_eval.export_chrome_trace(f"trace_eval_{args.gpu}.json")
    print(profile_train)
    print(profile_train.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    print(profile_eval.key_averages().table(sort_by=sort_by_keyword, row_limit=10))


if __name__ == "__main__":
    args = get_profile_argparse().parse_args()
    main(args)
