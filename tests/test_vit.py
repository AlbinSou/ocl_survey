#!/usr/bin/env python3
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torchvision.transforms import ToTensor

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import OnlineER_ACE
from avalanche.models.dynamic_modules import DynamicModule
from avalanche.models import SlimResNet18
from toolkit.utils import create_default_args, set_seed
from vit_pytorch import SimpleViT

def erace_scifar100(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "mem_size": 2000,
            "lr": 0.1,
            "train_mb_size": 10,
            "seed": 0,
            "batch_size_mem": 10,
            "train_passes": 1,
        },
        override_args
    )
    set_seed(args.seed)
    fixed_class_order = np.arange(100)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    unique_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
            ),
        ]
    )

    scenario = SplitCIFAR100(
        20,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=fixed_class_order,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
        train_transform=unique_transform,
        eval_transform=unique_transform,
    )

    input_size = (3, 32, 32)
    model = SimpleViT(image_size=input_size[1], patch_size=4,
            num_classes=100,
            dim=192,
            depth=4,
            heads=3,
            mlp_dim=512)
    print(sum([p.numel() for p in model.parameters()]))
    #model = SlimResNet18(100)
    optimizer = SGD(model.parameters(), lr=args.lr)
    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]
    training_metrics = []
    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
    ]
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )
    plugins = []

    cl_strategy = OnlineER_ACE(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        eval_mb_size=64,
        mem_size=args.mem_size,
        batch_size_mem=args.batch_size_mem,
        train_passes=args.train_passes,
    )


    batch_streams = scenario.streams.values()

    for t, experience in enumerate(scenario.train_stream):

        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=args.train_mb_size,
            access_task_boundaries=False,
        )

        cl_strategy.train(
            ocl_scenario.train_stream,
            num_workers=0,
            drop_last=True,
        )

        cl_strategy.eval(scenario.test_stream[: t + 1])
    results = cl_strategy.eval(scenario.test_stream)
    return results


if __name__ == "__main__":
    res = erace_scifar100()
    print(res)
