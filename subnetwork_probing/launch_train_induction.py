import subprocess
import numpy as np
import shlex
import random


def main(testing=False):
    # Base NLL:  0.9099822044372559
    # Reset NLL:  11.174062728881836
    # Reset KL:  10.114195823669434
    # All the losses are of order 10, so we can use the same scale

    regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )
    seed = random.randint(0, 2**31 - 1)

    i = 0
    for reset_subject in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["nll", "kl_div", "match_nll"]:
                for lambda_reg in [0.01] if testing else regularization_params:
                    command = [
                        "python",
                        "code/train_induction.py"
                        if testing
                        else "/Automatic-Circuit-Discovery/subnetwork-probing/code/train_induction.py",
                        "--task=induction",
                        f"--lambda-reg={lambda_reg:.3f}",
                        f"--wandb-name=agarriga-sp-{i:03d}",
                        "--wandb-project=induction-sp-replicate",
                        "--wandb-entity=remix_school-of-rock",
                        "--wandb-group=reset-with-nll-21",
                        f"--device=cuda",
                        f"--epochs={1 if testing else 10000}",
                        f"--zero-ablation={zero_ablation}",
                        f"--reset-subject={reset_subject}",
                        f"--seed={seed}",
                        f"--loss-type={loss_type}",
                        f"--num-examples={1 if testing else 50}",
                        f"--seq-len=300",
                        f"--n-loss-average-runs={1 if testing else 20}",
                        "--wandb-dir=/training/sp-induction",  # If it doesn't exist wandb will use /tmp
                        "--wandb-mode=online",
                    ]
                    command_str = shlex.join(command)
                    if testing:
                        print("Running", command_str)
                        # out = subprocess.run(command, check=True, capture_output=True)
                        print("Output:", out.stdout.decode("utf-8"))
                        continue

                    print("Launching", command_str)
                    subprocess.run(
                        [
                            "ctl",
                            "job",
                            "run",
                            f"--name=agarriga-sp-{i:03d}",
                            "--shared-host-dir-slow-tolerant",
                            "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.8",
                            "--cpu=4",
                            "--gpu=1",
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


if __name__ == "__main__":
    main()
