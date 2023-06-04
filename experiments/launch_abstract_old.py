from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import numpy as np
import random

def main(use_kubernetes: bool, testing: bool, CPU: int = 4):
    task = "docstring"
    reset_network = 0
    kwargses = [
        {"threshold": 0.067, "metric": "docstring_metric"},
        {"threshold": 0.005, "metric": "kl_div"},
        {"threshold": 0.095, "metric": "kl_div"},
    ]

    seed = 1495006036
    random.seed(seed)

    for kwargs in kwargses:
        wandb_identifier = WandbIdentifier(
            run_name=f"docstring_kl_{kwargs['threshold']:.5f}",
            group_name="default",
            project="acdc-abstract",
        )
        command = [
            "python",
            "acdc/main.py",
            f"--task={task}",
            f"--threshold={kwargs['threshold']:.5f}",
            "--using-wandb",
            "--wandb-entity-name=remix_school-of-rock",
            f"--wandb-run-name={wandb_identifier.run_name}",
            f"--wandb-project-name={wandb_identifier.project}",
        ]
        launch(
            [command],
            name=wandb_identifier.run_name,
            job=None
            if not use_kubernetes
            else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:abstract-0.0", cpu=CPU, gpu=1),
            check_wandb=wandb_identifier,
            just_print_commands=False,
        )


if __name__ == "__main__":
    main(use_kubernetes=True, testing=False)
