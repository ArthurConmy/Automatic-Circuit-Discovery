#!/usr/bin/env python3
import warnings
from experiments.launcher import KubernetesJob, launch
import subprocess
import numpy as np
import random
from typing import List

MODE = "nodes" # or "edges"
# TASKS = ["ioi", "docstring", "greaterthan"]
TASKS = ["tracr-reverse", "tracr-proportion"]

ADRIA_MODE = False

warnings.warn("Several edited sections so Arthur can get results")

METRICS_FOR_TASK = {
    # "ioi": ["kl_div", "logit_diff"],
    # "tracr-reverse": ["kl_div", "l2"],
    # "tracr-proportion": ["kl_div", "l2"],
    # "induction": ["kl_div", "nll"],
    # "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}

def main():
    commands = []
    for alg in ["16h", "sp", "acdc"]:
        for reset_network in [0]: # [0, 1]:
            for zero_ablation in [1]: # [0, 1]:
                if alg == "16h" and zero_ablation:
                    continue  # TODO remove

                for task in TASKS:
                    for metric in METRICS_FOR_TASK[task]:

                        if metric == "kl_div": 
                            continue

                        thing = [
                            task.lower(),
                            str(reset_network).lower(),
                            metric.lower(),
                            alg.lower(),
                            MODE.lower(),
                        ]
                        print(thing)
                        if thing != [
                            "tracr-reverse",
                            "0",
                            "l2",
                            "sp",
                            "nodes",
                        ]:
                            continue

                        command = [
                            "python",
                            "notebooks/roc_plot_generator.py",
                            f"--task={task}",
                            f"--reset-network={reset_network}",
                            f"--metric={metric}",
                            f"--alg={alg}",
                            f"--mode={MODE}",
                        ]
                        if zero_ablation:
                            command.append("--zero-ablation")
                        commands.append(command)

    if ADRIA_MODE:
        launch(commands, name="plots", job=None, synchronous=False)

    else: # don't do all the things async...
        for command in commands:
            print(" ".join(command))
            try:
                subprocess.run(command, check=True)
            except Exception as e:
                print(e, "was a failure")

if __name__ == "__main__":
    main()
