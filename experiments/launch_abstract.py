from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

def main(use_kubernetes: bool, testing: bool, CPU: int = 4):
    testing = True
    task = "docstring"
    reset_network = 0
    kwargses = [
        {"threshold": 0.067, "metric": "docstring_metric"},
        {"threshold": 0.005, "metric": "kl_div"},
        {"threshold": 0.095, "metric": "kl_div"},
    ]

    seed = 1495006036
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name="abstract-{i:05d}",
        group_name="abstract",
        project="acdc",
    )

    commands: List[List[str]] = []
    for kwargs in kwargses:
        command = [
            "python",
            "acdc/main.py",
            f"--task={task}",
            f"--threshold={kwargs['threshold']:.5f}",
            "--using-wandb",
            f"--wandb-run-name={wandb_identifier.run_name.format(i=len(commands))}",
            f"--wandb-group-name={wandb_identifier.group_name}",
            f"--wandb-project-name={wandb_identifier.project}",
            "--device=cuda",
            f"--torch-num-threads={CPU}",
            f"--reset-network={reset_network}",
            f"--seed={random.randint(0, 2**32 - 1)}",
            f"--metric={kwargs['metric']}",
            "--wandb-dir=/root/.cache/huggingface/tracr-training/acdc",  # If it doesn't exist wandb will use /tmp
            f"--wandb-mode=online",
            f"--max-num-epochs={1 if testing else 40_000}",
        ]
        commands.append(command)

    launch(
        commands,
        name="acdc-docstring-abstract",
        job=None
        if not use_kubernetes
        else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.1", cpu=CPU, gpu=1),
        check_wandb=wandb_identifier,
        just_print_commands=False,
    )


if __name__ == "__main__":
    main(use_kubernetes=True, testing=False)
