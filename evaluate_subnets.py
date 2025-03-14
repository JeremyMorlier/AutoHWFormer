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

import torch
import onnx
from zigzag import api

import utils
from args import get_eval_argparse
from logger import Logger

from libAutoFormer.config import cfg, update_config_from_file
from libAutoFormer.samplers import RASampler
from datasets.autoformer_datasets import build_dataset
from model.supernet_transformer import Vision_TransformerSuper
from references import MetricLogger, SmoothedValue
from train_supernet import sample_configs, evaluate, get_dataloaders

def evaluate_hardware(model, mapping_path, accelerator_path, output_dir, opt="energy") :

    # Generate ONNX model from Pytorch
    onnx_path = os.path.join(output_dir, "temp.onnx")
    inputs = (torch.rand(1, 3, 224, 224))
    onnx_model = torch.onnx.export(model, inputs, onnx_path)
    inferred_model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path), strict_mode=True)
    onnx.save(inferred_model, onnx_path)

    energy, latency, cme = api.get_hardware_performance_zigzag(workload=onnx_path, accelerator=accelerator_path, mapping=mapping_path, opt=opt)

    return {"energy": energy, "latency": latency, "cme": cme}

def main(args) :
    utils.init_distributed_mode(args)
    utils.init_signal_handler()

    device = torch.device(args.device)
    print(device)
    update_config_from_file(args.cfg)

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
            'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}
    
    if utils.is_main_process() :
        # similar API to wandb except mode and log_dir
        logger = Logger(project_name="HWAutoFormerEval",
                run_name=args.name,
                tags=["test"],
                resume=True,
                args=args,
                mode=args.logger,
                log_dir=args.output_dir)

    _, data_loader_val = get_dataloaders(args)
        
    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)

    model.to(device)
    # Parallelize the model using Distributed Data Parallel (DDP)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if utils.is_main_process() :
        logger.log({"nparams": n_parameters, "cuda_mem_allocated": torch.cuda.memory_allocated()})
    
    # Load trained supernet
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # Evaluate on  performance and hardware characteristics
    for k in range(0, args.n_models) :
        test_stats = evaluate(data_loader_val, model, device, choices=choices,  mode = args.mode, retrain_config=None)
        # for parameter in model.parameters() :
        #     parameter = parameter.detach()
        model_without_ddp.patch_embed_super.sampled_weight = model_without_ddp.patch_embed_super.sampled_weight.detach()
        model_without_ddp.patch_embed_super.sampled_bias = model_without_ddp.patch_embed_super.sampled_bias.detach()
        for block in model_without_ddp.blocks :
            # print(block.attn_layer_norm.samples)
            block.attn_layer_norm.samples['weight'] = block.attn_layer_norm.samples['weight'].detach()
            block.attn_layer_norm.samples['bias'] = block.attn_layer_norm.samples['bias'].detach()
        hardware_stats = evaluate_hardware(model, args.mapping, args.accelerator, args.output_dir)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}% ZigZag energy {hardware_stats['energy']:.1f}, ZigZag Latency {hardware_stats['latency']:.1f}")

if __name__ == "__main__" :
    parser = get_eval_argparse()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)