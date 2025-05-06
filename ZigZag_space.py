import os
import math
import random
import copy
import json
from tqdm.contrib.concurrent import process_map
import multiprocessing
import time
import yaml
from pathlib import Path
import argparse

import torch

from zigzag import api
from stream.api import optimize_allocation_ga

import onnx
from onnx import shape_inference
from onnxsim import simplify

from train_supernet import sample_configs
from model.transformer_onnx import SuperNet
from libAutoFormer.config import cfg, update_config_from_file
from hardware.stream_hardware_generator import stream_edge_tpu, stream_edge_tpu_core, stream_edge_tpu_mapping, to_yaml
from hardware.zigzag_hardware_generator import zigzag_edge_tpu_hardware, zigzag_edge_tpu_mapping
import logging

# # Disable ZigZag Logging and Pytorch Warnings
# logging.captureWarnings(True)
# logging.disable(logging.CRITICAL)


def argparser(add_help=True) :
    parser = argparse.ArgumentParser(description="Evaluate the hardware performance of the neural network space", add_help=add_help)
    parser.add_argument("--path", type=str, default="temp", help="Path to store temporary and final files")
    parser.add_argument("--tool", type=str, default="stream", help="tool to use (stream or zigzag)")
    return parser
    

def evaluate_performance(config) :

    time0 = time.time()
    result = {}

    # Extract config and constants
    model_config = config["model_config"]
    hardware_config = config["hardware_config"]
    mapping_config = config["mapping_config"]

    if config["mode"] == "stream" :
        modes = ["fused"]
        layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    elif config["mode"] == "zigzag" :
        modes = ["EDP"]
    else : 
        return None
    
    # Multiprocessing and paths
    id = multiprocessing.current_process().name
    id = id.split("-")[-1]

    onnx_name = "model.onnx"
    process_path = os.path.join(config["path"], id)
    onnx_path = os.path.join(process_path, onnx_name)
    inferred_path = os.path.join(process_path, "inferred_" + onnx_name)
    mapping_path = os.path.join(process_path, "mapping.yaml")
    result_path = os.path.join(process_path, "result.txt")

    core_path = os.path.join(process_path, "core.yaml")
    soc_path = os.path.join(process_path, "chip.yaml")
    Path(process_path).mkdir(parents=True, exist_ok=True)


    # Create Pytorch Model and convert to ONNX
    torch_model = SuperNet(
        image_size=224,
        patch_size=8,
        num_layers=model_config["layer_num"],
        num_headss=model_config["num_heads"],
        hidden_dims=model_config["embed_dim"],
        mlp_ratios=model_config["mlp_ratio"],
    )

    n_parameters = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    result["n_parameters"] = n_parameters

    inputs = torch.rand(1, 3, 224, 224)
    torch.onnx.export(torch_model, inputs, onnx_path)
    inferred_model, check = simplify(onnx.load(onnx_path))
    onnx.save(inferred_model, inferred_path)

    if config["mode"] == "stream" :
        # Generate Hardware and Mapping Config
        pooling_core_path = os.path.abspath("inputs/stream/examples/hardware/cores/pooling.yaml")
        simd_core_path = os.path.abspath("inputs/stream/examples/hardware/cores/pooling.yaml")
        offchip_core_path = os.path.abspath("inputs/stream/examples/hardware/cores/pooling.yaml")
        core = stream_edge_tpu_core(hardware_config["n_SIMDS"], hardware_config["n_computes_lanes"], hardware_config["PE_Memory"], hardware_config["register_file_size"])
        soc = stream_edge_tpu(hardware_config["xPE"], hardware_config["yPE"], os.path.abspath(core_path), [pooling_core_path, simd_core_path], offchip_core_path, 32, 0)
        mapping = stream_edge_tpu_mapping(hardware_config["xPE"], hardware_config["yPE"], ["pooling.yaml", "simd.yaml"])
        to_yaml(core, core_path)
        to_yaml(soc, soc_path)
        to_yaml(mapping, mapping_path)
        result["core"] = core
        result["soc"] = soc
    elif config["mode"] == "zigzag" :
        soc = zigzag_edge_tpu_hardware(hardware_config["xPE"], hardware_config["yPE"], hardware_config["n_SIMDS"], hardware_config["n_computes_lanes"], hardware_config["PE_Memory"], hardware_config["register_file_size"])
        mapping = zigzag_edge_tpu_mapping()
        to_yaml(soc, soc_path)
        to_yaml(mapping, mapping_path)
        result["soc"] = soc

    time1 = time.time()
    for mode in modes:
        result[mode] = {}
        try:
            if config["mode"] == "stream" :
                scme = optimize_allocation_ga(
                    hardware=soc_path,
                    workload=inferred_path,
                    mapping=mapping_path,
                    mode=mode,
                    layer_stacks=layer_stacks,
                    nb_ga_generations=4,
                    nb_ga_individuals=4,
                    experiment_id=id,
                    output_path=process_path,
                    skip_if_exists=False,
                )
                result[mode]["scme"] = vars(scme)
                result[mode]["energy"] = scme["energy"]
                result[mode]["latency"] = scme["latency"]
            elif config["mode"] == "zigzag" :
                energy, latency, cme = api.get_hardware_performance_zigzag(
                    workload=inferred_path,
                    accelerator=soc_path,
                    mapping=mapping_path,
                    opt=mode,
                )
                result[mode]["energy"] = energy
                result[mode]["latency"] = latency
        except Exception as e:
            print(e)
            result[mode]["energy"] = 0
            result[mode]["latency"] = 0
        # result[opt]["cme"] = cme
    time2 = time.time()

    result["preprocess"] = time1 - time0
    result["process"] = time2 - time1

    with open(result_path, "a") as file:
        json.dump(result, file)
        json.dump(config, file)
        file.write("\n")
    return True

def sample_hardware_configs(choices) :
    hardware_config = {"n_SIMDS":random.choice(choices["n_SIMDS"]),
                        "n_computes_lanes":random.choice(choices["n_computes_lanes"]), 
                        "PE_Memory": random.choice(choices["PE_Memory"]),
                        "register_file_size": random.choice(choices["register_file_size"]),
                        "xPE": random.choice(choices["xPE"]),
                        "yPE": random.choice(choices["yPE"])}
    return hardware_config

class Config_Generator:
    def __init__(self, max_iter, nn_choices, hw_choices, mapping_config, hardware_config, path, mode="zigzag"):
        self.max_iter = max_iter
        self.i = 0
        self.nn_choices = nn_choices
        self.hw_choices = hw_choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config
        self.path = path
        self.mode = mode

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["model_config"] = sample_configs(self.nn_choices)
            config["hardware_config"] = sample_hardware_configs(self.hw_choices)
            config["mapping_config"] = self.mapping_config
            config["path"] = self.path
            config["mode"] = self.mode
            return config
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_iter


if __name__ == "__main__" :


    args = argparser().parse_args()
    # accelerator_path = "inputs/hardware/tpu_like.yaml"
    # mapping_path = "inputs/stream/examples/mapping/tpu_like.yaml"

    update_config_from_file("config/supernet-B1.yaml")
    nn_choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }
    hw_choices = {
        "n_SIMDS": [16, 32, 64, 128],
        "n_computes_lanes": [1, 2, 4, 8], 
        "PE_Memory": [int(int(element*1024*1024*8)) for element in [0.5, 1, 2, 3, 4]],
        "register_file_size": [int(int(element*1024*8)) for element in [8, 16, 32, 48, 64]],
        "xPE": [1, 2, 4, 6, 8],
        "yPE": [1, 2, 4, 6, 8],
    }

    num_tasks = 100000
    num_workers = min(num_tasks, os.cpu_count() + 4)
    num_workers = 1
    chunksize = math.ceil(num_tasks / num_workers)

    chunksize = 1

    # test_config = {
    #     "path":"temp",
    #     "mode":"zigzag",
    #     "model_config":sample_configs(nn_choices),
    #     "hardware_config":{"n_SIMDS":16,"n_computes_lanes":64 ,"PE_Memory":2*1024*1024*8 ,"register_file_size":16*1024*8, "xPE":4,"yPE":4},
    #     "mapping_config":None

    # }
    # evaluate_performance(test_config)
    config_generator = Config_Generator(num_tasks, nn_choices, hw_choices, mapping_config=None, hardware_config=None, path=args.path)
    config_iterator = iter(config_generator)
    r = process_map(
        evaluate_performance,
        config_iterator,
        max_workers=num_workers,
        chunksize=chunksize,
    )

    print(r)
