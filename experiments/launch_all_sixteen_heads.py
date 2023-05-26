from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["greaterthan"],  # "kl_div",
}


CPU = 2

def main(TASKS: list[str], job: KubernetesJob, name: str, group_name: str):
    seed = 1259281515
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name=f"{name}-{{i:05d}}",
        group_name=group_name,
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
                        f"--torch-num-threads={CPU}",
                        "--wandb-dir=/root/.cache/huggingface/tracr-training/16heads",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode=online",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)


    launch(
        commands,
        name=wandb_identifier.run_name,
        job=job,
        check_wandb=wandb_identifier,
        just_print_commands=False,
        synchronous=True,
    )


if __name__ == "__main__":
    main(
        # ["ioi", "greaterthan", "induction", "docstring"],
        ["tracr-reverse"],
        KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.7.1", cpu=CPU, gpu=0),
        "16h-redo",
        group_name="sixteen-heads",
    )
    # main(
    #     ["tracr-reverse", "tracr-proportion"],
    #     KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.1", cpu=4, gpu=0),
    #     "16h-tracr",
    # )
