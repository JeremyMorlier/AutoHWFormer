import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from datasets.autoformer_datasets import build_dataset
import utils
from train_supernet import evaluate
from model.supernet_transformer import Vision_TransformerSuper
import argparse
import os
import yaml
from libAutoFormer.config import cfg, update_config_from_file

from args import get_evolutionary_argsparse


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return (
        depth,
        list(cand_tuple[1 : depth + 1]),
        list(cand_tuple[depth + 1 : 2 * depth + 1]),
        cand_tuple[-1],
    )


class EvolutionSearcher(object):
    def __init__(
        self,
        args,
        device,
        model,
        model_without_ddp,
        choices,
        val_loader,
        test_loader,
        output_dir,
    ):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices

    def save_checkpoint(self):
        info = {}
        info["top_accuracies"] = self.top_accuracies
        info["memory"] = self.memory
        info["candidates"] = self.candidates
        info["vis_dict"] = self.vis_dict
        info["keep_top_k"] = self.keep_top_k
        info["epoch"] = self.epoch
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch)
        )
        torch.save(info, checkpoint_path)
        print("save checkpoint to", checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info["memory"]
        self.candidates = info["candidates"]
        self.vis_dict = info["vis_dict"]
        self.keep_top_k = info["keep_top_k"]
        self.epoch = info["epoch"]

        print("load checkpoint from", self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if "visited" in info:
            return False
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config["layer_num"] = depth
        sampled_config["mlp_ratio"] = mlp_ratio
        sampled_config["num_heads"] = num_heads
        sampled_config["embed_dim"] = [embed_dim] * depth
        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info["params"] = n_parameters / 10.0**6

        if info["params"] > self.parameters_limits:
            print("parameters limit exceed")
            return False

        if info["params"] < self.min_parameters_limits:
            print("under minimum parameters limit")
            return False

        print("rank:", utils.get_rank(), cand, info["params"])
        eval_stats = evaluate(
            self.val_loader,
            self.model,
            self.device,
            amp=self.args.amp,
            mode="retrain",
            retrain_config=sampled_config,
        )
        test_stats = evaluate(
            self.test_loader,
            self.model,
            self.device,
            amp=self.args.amp,
            mode="retrain",
            retrain_config=sampled_config,
        )

        info["acc"] = eval_stats["acc1"]
        info["test_acc"] = test_stats["acc1"]

        info["visited"] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print("select ......")
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ["mlp_ratio", "num_heads"]
        depth = random.choice(self.choices["depth"])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))

        cand_tuple.append(random.choice(self.choices["embed_dim"]))
        return tuple(cand_tuple)

    def get_random(self, num):
        print("random select ........")
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print("random {}/{}".format(len(self.candidates), num))
        print("random_num = {}".format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print("mutation ......")
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()

            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices["depth"])

                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [
                        random.choice(self.choices["mlp_ratio"])
                        for _ in range(new_depth - depth)
                    ]
                    num_heads = num_heads + [
                        random.choice(self.choices["num_heads"])
                        for _ in range(new_depth - depth)
                    ]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]

                depth = new_depth
            # mlp_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    mlp_ratio[i] = random.choice(self.choices["mlp_ratio"])

            # num_heads

            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.choices["num_heads"])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.choices["embed_dim"])

            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print("mutation {}/{}".format(len(res), mutation_num))

        print("mutation_num = {}".format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print("crossover ......")
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print("crossover {}/{}".format(len(res), crossover_num))

        print("crossover_num = {}".format(len(res)))
        return res

    def search(self):
        print(
            "population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}".format(
                self.population_num,
                self.select_num,
                self.mutation_num,
                self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num,
                self.max_epochs,
            )
        )

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print("epoch = {}".format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[x]["acc"],
            )
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]["acc"]
            )

            print(
                "epoch = {} : top {} result".format(
                    self.epoch, len(self.keep_top_k[50])
                )
            )
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print(
                    "No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, params = {}".format(
                        i + 1,
                        cand,
                        self.vis_dict[cand]["acc"],
                        self.vis_dict[cand]["test_acc"],
                        self.vis_dict[cand]["params"],
                    )
                )
                tmp_accuracy.append(self.vis_dict[cand]["acc"])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob
            )
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()


def main(args):
    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        f.write(args_text)
    # fix the seed for reproducibility

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    dataset_val, args.nb_classes = build_dataset(
        is_train=False, args=args, folder_name="subImageNet"
    )
    dataset_test, _ = build_dataset(is_train=False, args=args, folder_name="val")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
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
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(2 * args.batch_size),
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(2 * args.batch_size),
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating SuperVisionTransformer")
    print(cfg)
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
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        print("resume from checkpoint: {}".format(args.resume))
        model_without_ddp.load_state_dict(checkpoint["model"])

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    t = time.time()
    searcher = EvolutionSearcher(
        args,
        device,
        model,
        model_without_ddp,
        choices,
        data_loader_val,
        data_loader_test,
        args.output_dir,
    )

    searcher.search()

    print("total searching time = {:.2f} hours".format((time.time() - t) / 3600))


if __name__ == "__main__":
    parser = get_evolutionary_argsparse()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
