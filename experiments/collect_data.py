import os
from pathlib import Path
from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import shlex
import random

TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main(alg: str, task: str, job: KubernetesJob, testing: bool = False):
    OUT_RELPATH = Path(".cache") / "plots_data"
    OUT_HOME_DIR = Path(os.environ["HOME"]) / OUT_RELPATH
    assert OUT_HOME_DIR.exists()

    OUT_DIR = Path("/root") / OUT_RELPATH

    seed = 1233778640
    random.seed(seed)

    commands = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for metric in METRICS_FOR_TASK[task]:
                command = [
                    "python",
                    "notebooks/roc_plot_generator.py",
                    f"--task={task}",
                    f"--reset-network={reset_network}",
                    f"--metric={metric}",
                    f"--alg={alg}",
                    f"--device={'cpu' if testing or not job.gpu else 'cuda'}",
                    f"--torch-num-threads={job.cpu}",
                    f"--out-dir={OUT_DIR}",
                    f"--seed={random.randint(0, 2**31-1)}",
                ]
                if zero_ablation:
                    command.append("--zero-ablation")
                commands.append(command)

    launch(
        commands,
        name="collect_data",
        job=job,
        synchronous=True,
        just_print_commands=False,
        check_wandb=WandbIdentifier(f"agarriga-collect-{alg}-{task[-5:]}-{{i:05d}}", "collect", "acdc"),
    )


if __name__ == "__main__":
    for alg in ["acdc", "16h", "sp"]:
        for task in TASKS:
            main(
                alg,
                task,
                KubernetesJob(
                    container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.6.7",
                    cpu=4,
                    gpu=0 if task.startswith("tracr") or alg != "acdc" else 1,
                    mount_training=False,
                ),
                testing=False,
            )
