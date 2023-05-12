from experiments.launcher import KubernetesJob, launch
import numpy as np

CPU = 4


def main(testing: bool):
    thresholds = 10 ** np.linspace(-2, 0.5, 21)
    seed = 424671755

    commands: list[list[str]] = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div", "nll", "match_nll"]:
                for threshold in [1.0] if testing else thresholds:
                    command = [
                        "python",
                        "acdc/main.py" if testing else "/Automatic-Circuit-Discovery/acdc/main.py",
                        "--task=induction",
                        f"--threshold={threshold:.5f}",
                        "--using-wandb",
                        f"--wandb-run-name=agarriga-acdc-{len(commands):03d}",
                        "--wandb-group-name=adria-induction-3",
                        f"--device=cpu",
                        f"--reset-network={reset_network}",
                        f"--seed={seed}",
                        f"--metric={loss_type}",
                        f"--torch-num-threads={CPU}",
                        "--wandb-dir=/training/acdc",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode={'offline' if testing else 'online'}",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)

    launch(
        commands,
        name="acdc-induction",
        job=None
        if testing
        else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.8", cpu=CPU, gpu=0),
    )


if __name__ == "__main__":
    main(testing=False)
