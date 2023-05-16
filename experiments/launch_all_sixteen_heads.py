from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

TASKS = ["ioi", "tracr-reverse", "tracr-proportion", "greaterthan", "induction", "docstring"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main():
    seed = 1259281515
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name="agarriga-16heads-{i:05d}",
        group_name="sixteen-heads",
        project="acdc")

    commands: List[List[str]] = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:
                    command = [
                        "python",
                        "experiments/launch_sixteen_heads.py",
                        f"--task={task}",
                        f"--wandb-run-name={wandb_identifier.run_name.format(i=len(commands))}",
                        f"--wandb-group={wandb_identifier.group_name}",
                        f"--wandb-project={wandb_identifier.project}",
                        f"--device=cuda",
                        f"--reset-network={reset_network}",
                        f"--seed={random.randint(0, 2**32 - 1)}",
                        f"--metric={metric}",
                        "--wandb-dir=/training/16heads",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode=online",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)

    launch(commands, name="16heads", job=None, check_wandb=None)


if __name__ == "__main__":
    main()
