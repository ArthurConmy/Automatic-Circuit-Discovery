#!/usr/bin/env python3
from experiments.launcher import KubernetesJob, launch
import numpy as np
import random
from typing import List


TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main():
    commands = []
    for alg in ["16h", "sp", "acdc"]:
        for reset_network in [0, 1]:
            for zero_ablation in [0, 1]:
                for task in TASKS:
                    for metric in METRICS_FOR_TASK[task]:
                        command = [
                            "python",
                            "notebooks/roc_plot_generator.py",
                            f"--task={task}",
                            f"--reset-network={reset_network}",
                            f"--metric={metric}",
                            f"--alg={alg}",
                        ]
                        if zero_ablation:
                            command.append("--zero-ablation")
                        commands.append(command)

    launch(commands, name="plots", job=None, synchronous=False)


if __name__ == "__main__":
    main()
