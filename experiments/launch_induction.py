import subprocess
import numpy as np
import shlex
import random


def main(testing=False, use_kubernetes=False):
    thresholds = 10 ** np.linspace(-2, 0.5, 21)
    seed = random.randint(0, 2**31 - 1)

    to_wait = []
    i = 0
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div"]:
                for threshold in [1.0] if testing else thresholds:
                    command = [
                        "python",
                        "acdc/main.py"
                        if testing or not use_kubernetes
                        else "/Automatic-Circuit-Discovery/acdc/main.py",
                        "--task=induction",
                        f"--threshold={threshold:.5f}",
                        "--using-wandb",
                        f"--wandb-run-name=agarriga-acdc-{i:03d}",
                        "--wandb-group-name=adria-induction-2",
                        f"--device={'cpu' if testing else 'cuda'}",
                        f"--reset-network={reset_network}",
                        f"--seed={seed}",
                        f"--metric={loss_type}",
                        "--torch-num-threads=1",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

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
                                f"--name=agarriga-acdc-{i:03d}",
                                "--shared-host-dir-slow-tolerant",
                                "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.6",
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

    print("to wait", to_wait)
    if not testing and not use_kubernetes:
        for process_to_wait in to_wait:
            process_to_wait.wait()


if __name__ == "__main__":
    main(use_kubernetes=True)
