from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random
from typing import List

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main(TASKS: list[str], job: KubernetesJob, name: str, testing: bool, reset_networks: bool):
    NUM_SPACINGS = 5 if reset_networks else 21
    expensive_base_regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )

    if reset_networks:
        base_regularization_params = 10 ** np.linspace(-2, 1.5, NUM_SPACINGS)
    else:
        base_regularization_params = expensive_base_regularization_params

    wandb_identifier = WandbIdentifier(
        run_name=f"{name}-{{i:05d}}",
        group_name="reset-networks3",
        project="induction-sp-replicate")


    seed = 1507014021
    random.seed(seed)

    commands: List[List[str]] = []
    for reset_network in [int(reset_networks)]:
        for zero_ablation in [0, 1]:
            for task in TASKS:
                for metric in METRICS_FOR_TASK[task]:
                    if task.startswith("tracr"):
                        # Typical metric value range: 0.0-0.1
                        regularization_params = 10 ** np.linspace(-3, 0, 11)

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
                            regularization_params = base_regularization_params
                        elif metric == "greaterthan":
                            # Typical metric value range: -1.0 - 0.0
                            regularization_params = 10 ** np.linspace(-4, 2, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 100
                        seq_len = -1
                    elif task == "docstring":
                        seq_len = 41
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-10.0
                            regularization_params = expensive_base_regularization_params
                        elif metric == "docstring_metric":
                            # Typical metric value range: -1.0 - 0.0
                            regularization_params = 10 ** np.linspace(-4, 2, 21)
                        else:
                            raise ValueError("Unknown metric")
                        num_examples = 50
                    elif task == "ioi":
                        num_examples = 100
                        seq_len = -1
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-12.0
                            regularization_params = base_regularization_params
                        elif metric == "logit_diff":
                            # Typical metric value range: -0.31 -- -0.01
                            regularization_params = 10 ** np.linspace(-4, 2, NUM_SPACINGS)
                        else:
                            raise ValueError("Unknown metric")
                    elif task == "induction":
                        seq_len = 300
                        num_examples  = 50
                        if metric == "kl_div":
                            # Typical metric value range: 0.0-16.0
                            regularization_params = expensive_base_regularization_params
                        elif metric == "nll":
                            # Typical metric value range: 0.0-16.0
                            regularization_params = expensive_base_regularization_params
                        else:
                            raise ValueError("Unknown metric")
                    else:
                        raise ValueError("Unknown task")

                    for lambda_reg in [0.01] if testing else regularization_params:
                        command = [
                            "python",
                            "subnetwork_probing/train.py",
                            f"--task={task}",
                            f"--lambda-reg={lambda_reg:.3f}",
                            f"--wandb-name=agarriga-sp-{len(commands):05d}{'-optional' if task in ['induction', 'docstring'] else ''}",
                            "--wandb-project=induction-sp-replicate",
                            "--wandb-entity=remix_school-of-rock",
                            "--wandb-group=tracr-shuffled-redo",
                            f"--device={'cpu' if testing or not job.gpu else 'cuda'}",
                            f"--epochs={1 if testing else 10000}",
                            f"--zero-ablation={zero_ablation}",
                            f"--reset-subject={reset_network}",
                            f"--seed={random.randint(0, 2**32 - 1)}",
                            f"--loss-type={metric}",
                            f"--num-examples={6 if testing else num_examples}",
                            f"--seq-len={seq_len}",
                            f"--n-loss-average-runs={1 if testing else 20}",
                            "--wandb-dir=/training",  # If it doesn't exist wandb will use /tmp
                            f"--wandb-mode=offline",
                            f"--torch-num-threads={job.cpu}",
                        ]
                        commands.append(command)

    launch(
        commands,
        name="complete-spreadsheet",
        job=job,
        synchronous=True,
        check_wandb=wandb_identifier,
        just_print_commands=False,
    )

if __name__ == "__main__":
    # main(
    #     ["ioi", "greaterthan", "induction", "docstring"],
    #     KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.6", cpu=2, gpu=1),
    #     "sp-gpu",
    #     testing=False,
    #     reset_networks=True,
    # )
    main(
        ["tracr-reverse", "tracr-proportion"],
        KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.6", cpu=4, gpu=0, mount_training=True),
        "sp-tracr",
        testing=False,
        reset_networks=True,
    )
