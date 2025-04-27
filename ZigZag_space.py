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

import torch

from zigzag import api
from stream.api import optimize_allocation_ga

import onnx
from onnx import shape_inference

from train_supernet import sample_configs
from model.transformer_onnx import SuperNet
from libAutoFormer.config import cfg, update_config_from_file
from hardware_generator import edge_tpu, edge_tpu_core, edge_tpu_mapping, to_yaml
import logging

# Disable ZigZag Logging and Pytorch Warnings
logging.captureWarnings(True)
logging.disable(logging.CRITICAL)


def stream_performance(config) :
    time0 = time.time()

    # Extract config and constants
    result = {}
    opts = ["EDP"]
    modes = ["fused"]
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))


    model_config = config["model_config"]
    hardware_config = config["hardware_config"]
    mapping_config = config["mapping_config"]

    # Multiprocessing and paths
    id = multiprocessing.current_process().name
    id = id.split("-")[-1]

    onnx_name = "model.onnx"
    process_path = os.path.join("temp", id)
    onnx_path = os.path.join(process_path, onnx_name)
    inferred_path = os.path.join(process_path, "inferred_" + onnx_name)
    accelerator_path = os.path.join(process_path, "accelerator.yaml")
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
    # onnx_model = torch.onnx.dynamo_export(torch_model, args).save(path, external_data=True)
    torch.onnx.export(torch_model, inputs, onnx_path)
    inferred_model = shape_inference.infer_shapes(onnx.load(onnx_path), strict_mode=True)
    onnx.save(inferred_model, inferred_path)

    # Generate Hardware and Mapping Config
    core = edge_tpu_core(hardware_config["n_SIMDS"], hardware_config["n_computes_lanes"], hardware_config["PE_Memory"], hardware_config["register_file_size"])
    soc = edge_tpu(hardware_config["xPE"], hardware_config["yPE"], os.path.abspath(core_path), ["pooling.yaml", "simd.yaml"], "offchip.yaml", 32, 0)
    mapping = edge_tpu_mapping(hardware_config["xPE"], hardware_config["yPE"], ["pooling.yaml", "simd.yaml"])
    to_yaml(core, core_path)
    to_yaml(soc, soc_path)
    to_yaml(mapping, mapping_path)
    result["core"] = core
    result["soc"] = soc

    time1 = time.time()

    for mode in modes:
        result[mode] = {}
        try:
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


def zigzag_performance(config):
    time0 = time.time()
    # Extract config
    result = {}
    opts = ["EDP"]

    model_config = config["model_config"]
    hardware_config = config["hardware_config"]
    mapping_config = config["mapping_config"]

    # Multiprocessing and paths
    id = multiprocessing.current_process().name
    id = id.split("-")[-1]

    onnx_name = "model.onnx"
    process_path = os.path.join("/users/local/j20morli/temp", id)
    onnx_path = os.path.join(process_path, onnx_name)
    inferred_path = os.path.join(process_path, "inferred_" + onnx_name)
    accelerator_path = os.path.join(process_path, "accelerator.yaml")
    mapping_path = os.path.join(process_path, "mapping.yaml")
    result_path = os.path.join(process_path, "result.txt")
    Path(process_path).mkdir(parents=True, exist_ok=True)

    # Saves Mapping and Accelerator Configs for zigzag
    with open(accelerator_path, "w") as accelerator_file:
        yaml.safe_dump(hardware_config, accelerator_file, sort_keys=False)
    with open(mapping_path, "w") as mapping_file:
        yaml.safe_dump(mapping_config, mapping_file, sort_keys=False)

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
    # onnx_model = torch.onnx.dynamo_export(torch_model, args).save(path, external_data=True)
    torch.onnx.export(torch_model, inputs, onnx_path)
    inferred_model = shape_inference.infer_shapes(onnx.load(onnx_path), strict_mode=True)
    onnx.save(inferred_model, inferred_path)
    time1 = time.time()

    for opt in opts:
        result[opt] = {}
        try:
            energy, latency, cme = api.get_hardware_performance_zigzag(
                workload=inferred_path,
                accelerator=accelerator_path,
                mapping=mapping_path,
                opt=opt,
            )
            result[opt]["energy"] = energy
            result[opt]["latency"] = latency
        except Exception as e:
            print(e)
            result[opt]["energy"] = 0
            result[opt]["latency"] = 0
        # result[opt]["cme"] = cme
    time2 = time.time()

    result["preprocess"] = time1 - time0
    result["process"] = time2 - time1

    with open(result_path, "a") as file:
        json.dump(result, file)
        json.dump(config, file)
        file.write("\n")
    return True


def zigzag_sample_hardware_configs(choices, hardware_config):
    hw_config = copy.deepcopy(hardware_config)
    for key, item in choices.items():
        for key2, item2 in item.items():
            if isinstance(item2, dict):
                for key3, item3 in item2.items():
                    hardware_config[key][key2][key3] = random.choice(item3)
            else:
                hardware_config[key][key2] = random.choice(item2)
    return hw_config

def stream_sample_hardware_configs(choices) :
    hardware_config = {"n_SIMDS":random.choice(choices["n_SIMDS"]),
                        "n_computes_lanes":random.choice(choices["n_computes_lanes"]), 
                        "PE_Memory": random.choice(choices["PE_Memory"]),
                        "register_file_size": random.choice(choices["register_file_size"]),
                        "xPE": random.choice(choices["xPE"]),
                        "yPE": random.choice(choices["yPE"])}
    return hardware_config

class Config_Generator:
    def __init__(self, max_iter, nn_choices, hw_choices, mapping_config, hardware_config):
        self.max_iter = max_iter
        self.i = 0
        self.nn_choices = nn_choices
        self.hw_choices = hw_choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["model_config"] = sample_configs(self.nn_choices)
            config["hardware_config"] = stream_sample_hardware_configs(self.hw_choices)
            config["mapping_config"] = self.mapping_config
            return config
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_iter


if __name__ == "__main__":
    # accelerator_path = "inputs/hardware/tpu_like.yaml"
    # mapping_path = "inputs/stream/examples/mapping/tpu_like.yaml"

    update_config_from_file("config/supernet-B1.yaml")
    nn_choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }
    stream_hw_choices = {
        "n_SIMDS": [16, 32, 64, 128],
        "n_computes_lanes": [1, 2, 4, 8], 
        "PE_Memory": [element*1024*1024 for element in [0.5, 1, 2, 3, 4]],
        "register_file_size": [element*1024 for element in [8, 16, 32, 48, 64]],
        "xPE": [1, 2, 4, 6, 8],
        "yPE": [1, 2, 4, 6, 8],
    }
    zigzag_hw_choices = {
        "memories": {
            "rf_128B": {
                "size": [512, 1024, 2048],
            },
            "rf_2B": {
                "size": [4, 8, 16, 32],
            },
            "sram_2MB": {
                "size": [8388608, 16777216, 33554432],
            },
        },
        "operational_array": {"sizes": [[16, 16], [24, 24], [32, 32], [48, 48]]},
    }

    # with open(accelerator_path, "r") as accelerator_file:
    #     hardware_config = yaml.safe_load(accelerator_file)
    # with open(mapping_path, "r") as mapping_file:
    #     mapping_config = yaml.safe_load(mapping_file)

    num_tasks = 100000
    num_workers = min(num_tasks, 32, os.cpu_count() + 4)
    chunksize = math.ceil(num_tasks / num_workers)

    chunksize = 1

    config_generator = Config_Generator(num_tasks, nn_choices, stream_hw_choices, mapping_config=None, hardware_config=None)
    config_iterator = iter(config_generator)
    r = process_map(
        stream_performance,
        config_iterator,
        max_workers=num_workers,
        chunksize=chunksize,
    )

    print(r)
