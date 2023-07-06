from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}

CPU = 4

def main(TASKS: list[str], group_name: str, run_name: str, testing: bool, use_kubernetes: bool, reset_networks: bool, abs_value_threshold: bool, use_gpu: bool=True):
    NUM_SPACINGS = 5 if reset_networks else 21
    base_thresholds = 10 ** np.linspace(-4, 0, 21)

    seed = 486887094
    random.seed(seed)

    wandb_identifier = WandbIdentifier(
        run_name=run_name,
        group_name=group_name,
        project="acdc")

    commands: List[List[str]] = []
    for reset_network in [int(reset_networks)]:
        for zero_ablation in [0]:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:

                    if task.startswith("tracr"):
                        # Typical metric value range: 0.0-0.1
                        thresholds = 10 ** np.linspace(-5, -1, 21)

                        if task == "tracr-reverse":
                            num_examples = 6
                            seq_len = 5
                        elif task == "tracr-proportion":
                            num_examples = 50
                            seq_len = 5
                        else:
                            raise ValueError("Unknown task")

                    elif task == "greaterthan":
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-20
                            # thresholds = 10 ** np.linspace(-4, 0, NUM_SPACINGS)
                            thresholds = 10 ** np.linspace(-6, -4, 11)
                        elif metric == "greaterthan":
                            # Typical metric value range: -1.0 - 0.0
                            thresholds = 10 ** np.linspace(-3, -1, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 100
                        seq_len = -1
                    elif task == "docstring":
                        seq_len = 41
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-10.0
                            thresholds = base_thresholds
                        elif metric == "docstring_metric":
                            # Typical metric value range: -1.0 - 0.0
                            thresholds = 10 ** np.linspace(-4, 0, 21)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 50
                    elif task == "ioi":
                        num_examples = 100
                        seq_len = -1
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-12.0
                            thresholds = 10 ** np.linspace(-6, 0, 31)
                        elif metric == "logit_diff":
                            # Typical metric value range: -0.31 -- -0.01
                            thresholds = 10 ** np.linspace(-4, 0, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                    elif task == "induction":
                        seq_len = 300
                        num_examples  = 50
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-16.0
                            thresholds = base_thresholds
                        elif metric == "nll":
                            # Typical metric value range: 0.0-16.0
                            thresholds = base_thresholds
                        else:
                            raise ValueError("Unknown metric")
                    else:
                        raise ValueError("Unknown task")

                    for threshold in [1.0] if testing else thresholds:
                        command = [
                            "python",
                            "acdc/main.py",
                            f"--task={task}",
                            f"--threshold={threshold}",
                            "--using-wandb",
                            f"--wandb-run-name={wandb_identifier.run_name.format(i=len(commands))}",
                            f"--wandb-group-name={wandb_identifier.group_name}",
                            f"--wandb-project-name={wandb_identifier.project}",
                            f"--device={'cuda' if not testing else 'cpu'}" if "tracr" not in task else "--device=cpu",
                            f"--reset-network={reset_network}",
                            f"--seed={random.randint(0, 2**32 - 1)}",
                            f"--metric={metric}",
                            f"--torch-num-threads={CPU}",
                            "--wandb-dir=/root/.cache/huggingface/tracr-training/acdc",  # If it doesn't exist wandb will use /tmp
                            f"--wandb-mode=online",
                            f"--max-num-epochs={1 if testing else 40_000}",
                        ]
                        if zero_ablation:
                            command.append("--zero-ablation")
                        if abs_value_threshold:
                            command.append("--abs-value-threshold")
                        commands.append(command)

    launch(
        commands,
        name="acdc-spreadsheet",
        job=None
        if not use_kubernetes
        else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:181999f", cpu=CPU, gpu=int(use_gpu)),
        check_wandb=wandb_identifier,
        just_print_commands=False,
    )


if __name__ == "__main__":
    for reset_networks in [False]:
        main(
            TASKS=["ioi"],
            group_name="abs-value",
            run_name=f"agarriga-ioi-res{int(reset_networks)}-{{i:05d}}",
            testing=False,
            use_kubernetes=True,
            reset_networks=reset_networks,
            abs_value_threshold=True,
            use_gpu=True,
        )

# if __name__ == "__main__":
#     for reset_networks in [False, True]:
#         main(
#             ["ioi", "greaterthan", "induction", "docstring"],
#             "reset-networks-neurips",
#             "agarriga-tracr3-{i:05d}",
#             testing=False,
#             use_kubernetes=True,
#             reset_networks=True,
#         )
#         main(
#             ["induction"],
#             "adria-induction-3",
#             "agarriga-induction-{i:05d}",
#             testing=False,
#             use_kubernetes=True,
#             reset_networks=False,
#         )
