from experiments.launcher import KubernetesJob, launch
import numpy as np

CPU = 2


def main(testing: bool):
    regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )
    seed = 2083893729

    commands: list[list[str]] = []
    for reset_network in [0]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div", "docstring_metric", "docstring_stefan", "nll", "match_nll"]:
                for lambda_reg in [0.01] if testing else regularization_params:
                    command = [
                        "python",
                        "code/train_induction.py"
                        if testing
                        else "/Automatic-Circuit-Discovery/subnetwork-probing/code/train_induction.py",
                        "--task=docstring",
                        f"--lambda-reg={lambda_reg:.3f}",
                        f"--wandb-name=agarriga-sp-{len(commands):03d}",
                        "--wandb-project=induction-sp-replicate",
                        "--wandb-entity=remix_school-of-rock",
                        "--wandb-group=reset-with-nll-21",
                        f"--device=cuda",
                        f"--epochs={1 if testing else 10000}",
                        f"--zero-ablation={zero_ablation}",
                        f"--reset-subject={reset_network}",
                        f"--seed={seed}",
                        f"--loss-type={loss_type}",
                        f"--num-examples={1 if testing else 50}",
                        f"--seq-len=41",
                        f"--n-loss-average-runs={1 if testing else 20}",
                        "--wandb-dir=/training/sp",  # If it doesn't exist wandb will use /tmp
                        "--wandb-mode=online",
                    ]
                    commands.append(command)

    launch(
        commands,
        name="sp-docstring",
        job=None
        if testing
        else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.10", cpu=CPU, gpu=1),
    )


if __name__ == "__main__":
    main(testing=False)
