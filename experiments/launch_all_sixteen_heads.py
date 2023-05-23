from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import subprocess
import random
from typing import List

IS_ARTHUR = "arthurworkspace" in __file__

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["greaterthan"] if IS_ARTHUR else ["kl_div", "greaterthan"],
}

def main(TASKS: list[str], job: KubernetesJob, name: str):
    seed = 1259281515
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name=f"{name}-{{i:05d}}",
        group_name=f"{'arthur-' if IS_ARTHUR else ''}sixteen-heads",
        project="acdc")

    commands: List[List[str]] = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:
                    if "tracr" not in task:
                        if reset_network==0 and zero_ablation==0:
                            continue
                        if task in ["ioi", "induction"] and reset_network==0 and zero_ablation==1:
                            continue

                    command = [
                        "python",
                        "experiments/launch_sixteen_heads.py",
                        f"--task={task}",
                        f"--wandb-run-name={wandb_identifier.run_name.format(i=len(commands))}",
                        f"--wandb-group={wandb_identifier.group_name}",
                        f"--wandb-project={wandb_identifier.project}",
                        f"--device={'cuda' if job.gpu else 'cpu'}",
                        f"--reset-network={reset_network}",
                        f"--seed={random.randint(0, 2**32 - 1)}",
                        f"--metric={metric}",
                        "--wandb-dir=/training/16heads",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode=online",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)


    if IS_ARTHUR:
        for command in commands:
            print(" ".join(command))
            subprocess.run(command)

    else:
        launch(commands, name=wandb_identifier.run_name, job=job, check_wandb=wandb_identifier, just_print_commands=False)


if __name__ == "__main__":
    main(
        ["greaterthan"] if IS_ARTHUR else ["ioi", "greaterthan", "induction", "docstring"],
        KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.1", cpu=2, gpu=1),
        "16h-gpu",
    )

    if not IS_ARTHUR:
        main(
            ["tracr-reverse", "tracr-proportion"],
            KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.1", cpu=4, gpu=0),
            "16h-tracr",
        )
