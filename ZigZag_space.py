import os
import math
from tqdm.contrib.concurrent import process_map
import multiprocessing
import time
import yaml
from pathlib import Path
import torch
from zigzag import api
import onnx
from onnx import shape_inference

from train_supernet import sample_configs
from model.transformer_onnx import SuperNet
from libAutoFormer.config import cfg, update_config_from_file

import logging

# Disable ZigZag Logging and Pytorch Warnings
logging.captureWarnings(True)
logging.disable(logging.CRITICAL)


def zigzag_performance(config):
    time0 = time.time()
    # Extract config
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
    inputs = torch.rand(1, 3, 224, 224)
    # onnx_model = torch.onnx.dynamo_export(torch_model, args).save(path, external_data=True)
    torch.onnx.export(torch_model, inputs, onnx_path)
    inferred_model = shape_inference.infer_shapes(
        onnx.load(onnx_path), strict_mode=True
    )
    onnx.save(inferred_model, inferred_path)
    time1 = time.time()
    result = {}
    opts = ["energy", "latency", "EDP"]
    for opt in opts:
        result[opt] = {}
        energy, latency, cme = api.get_hardware_performance_zigzag(
            workload=inferred_path,
            accelerator=accelerator_path,
            mapping=mapping_path,
            opt=opt,
        )
        result[opt]["energy"] = energy
        result[opt]["latency"] = latency
        # result[opt]["cme"] = cme
    time2 = time.time()

    result["preprocess"] = time1 - time0
    result["process"] = time2 - time1
    return config, result


class Config_Generator:
    def __init__(self, max_iter, choices, mapping_config, hardware_config):
        self.max_iter = max_iter
        self.i = 0
        self.choices = choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["model_config"] = sample_configs(self.choices)
            config["hardware_config"] = self.hardware_config
            config["mapping_config"] = mapping_config
            return config
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_iter


if __name__ == "__main__":
    accelerator_path = "inputs/hardware/tpu_like.yaml"
    mapping_path = "inputs/mapping/tpu_like.yaml"

    update_config_from_file("config/supernet-B1.yaml")
    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    with open(accelerator_path, "r") as accelerator_file:
        hardware_config = yaml.safe_load(accelerator_file)
    with open(mapping_path, "r") as mapping_file:
        mapping_config = yaml.safe_load(mapping_file)

    # test_config = sample_configs(choices)

    # config = {"model_config":test_config, "hardware_config": hardware_config, "mapping_config": mapping_config}
    # new_config, result = zigzag_performance(config)
    # print(result["preprocess"], result["process"], result["energy"], result["latency"])

    num_tasks = 1000
    num_workers = min(num_tasks, 32, os.cpu_count() + 4)
    chunksize = math.ceil(num_tasks / num_workers)
    chunksize = 1

    config_generator = Config_Generator(
        num_tasks, choices, mapping_config, hardware_config
    )
    config_iterator = iter(config_generator)
    r = process_map(
        zigzag_performance,
        config_iterator,
        max_workers=num_workers,
        chunksize=chunksize,
    )

    print(r)
    # r2 = [int(element[-1]) for element in r]
    # print(list(set(r2)))
    # somme = 0
    # for element in list(set(r2)) :
    #     somme += r2.count(element)
    #     print("id ", element, " nb repetitions : ", r2.count(element))
    # print(somme, chunksize)
    # #print(list(set(r)))
    # # print(r)
