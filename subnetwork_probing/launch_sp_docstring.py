import subprocess
import numpy as np
import shlex
import random


def main(testing=False, use_kubernetes=False):
    regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )
    seed = random.randint(0, 2**31 - 1)

    to_wait = []
    i = 0
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
                        f"--wandb-name=agarriga-sp-{i:03d}",
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
                        "--wandb-dir=/training/sp-induction",  # If it doesn't exist wandb will use /tmp
                        "--wandb-mode=online",
                    ]

                    command_str = shlex.join(command)
                    print("Launching", command_str)
                    if testing or not use_kubernetes:
                        out = subprocess.Popen(
                            command, stdout=open(f"stdout_{i:03d}.txt", "w"), stderr=open(f"stderr_{i:03d}.txt", "w")
                        )
                        to_wait.append(out)

                    if not testing and use_kubernetes:
                        subprocess.run(
                            [
                                "ctl",
                                "job",
                                "run",
                                f"--name=agarriga-sp-docstring-{i:03d}",
                                "--shared-host-dir-slow-tolerant",
                                "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.10",
                                "--cpu=4",
                                "--gpu=0",
                                "--login",
                                "--wandb",
                                "--never-restart",
                                f"--command={command_str}",
                                "--working-dir=/Automatic-Circuit-Discovery",
                                "--shared-host-dir=/home/agarriga/.cache/huggingface",
                                "--shared-host-dir-mount=/root/.cache/huggingface",
                            ],
                            check=True,
                        )
                    i += 1

    print("to wait", to_wait)
    if testing or not use_kubernetes:
        for process_to_wait in to_wait:
            process_to_wait.wait()


if __name__ == "__main__":
    main(testing=True)
