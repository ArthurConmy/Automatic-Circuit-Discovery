from pathlib import Path
import subprocess
from typing import Optional, TextIO, List, Tuple
import numpy as np
import shlex
import dataclasses
import wandb
import os

cwd = str(os.getcwd())
IS_ARTHUR = cwd.startswith("/root") or "aconmy" in cwd or "arthur" in cwd
IS_ADRIA = not IS_ARTHUR
ON_HOFVARPNIR = IS_ADRIA or "aconmy" in cwd

@dataclasses.dataclass(frozen=True)
class KubernetesJob:
    container: str
    cpu: int
    gpu: int
    mount_training: bool=False

    def mount_training_options(self) -> List[str]:
        if not self.mount_training:
            return []
        return [
            "--volume-mount=/training",
            "--volume-name=agarriga-models-training",
        ]


@dataclasses.dataclass(frozen=True)
class WandbIdentifier:
    run_name: str
    group_name: str
    project: str


def launch(commands: List[List[str]], name: str, job: Optional[KubernetesJob] = None, check_wandb: Optional[WandbIdentifier]=None, ids_for_worker=range(0, 10000000), synchronous=True, just_print_commands=False):
    to_wait: List[Tuple[str, subprocess.Popen, TextIO, TextIO]] = []

    assert len(commands) <= 100_000, "Too many commands for 5 digits"

    print(f"Launching {len(commands)} jobs")
    for i, command in enumerate(commands):
        if i not in ids_for_worker:
            print(f"Skipping {name} because it's not my turn, {i} not in {ids_for_worker}")
            continue

        command_str = shlex.join(command)


        if check_wandb is not None:
            # HACK this is pretty vulnerable to duplicating work if the same run is launched in close succession,
            # it's more to be able to restart
            # api = wandb.Api()
            name = check_wandb.run_name.format(i=i)
            # if name in existing_names:
            #     print(f"Skipping {name} because it already exists")
            #     continue

            # runs = api.runs(path=f"remix_school-of-rock/{check_wandb.project}", filters={"group": check_wandb.group_name})
            # existing_names = existing_names.union({r.name for r in runs})
            # print("Runs that exist: ", existing_names)
            # if name in existing_names:
            #     print(f"Run {name} already exists, skipping")
            #     continue

        print("Launching", name, command_str)
        if just_print_commands:
            continue

        if job is None:
            if synchronous:
                out = subprocess.run(command)
                assert out.returncode == 0, f"Command return={out.returncode} != 0"
            else:
                base_path = Path(f"/tmp/{name}")
                base_path.mkdir(parents=True, exist_ok=True)
                stdout = open(base_path / f"stdout_{i:05d}.txt", "w")
                stderr = open(base_path / f"stderr_{i:05d}.txt", "w")
                out = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                to_wait.append((command_str, out, stdout, stderr))
        else:
            if "cuda" in command_str:
                assert job.gpu > 0
            else:
                assert job.gpu == 0

            # fpath = Path("/root/sleipnir/ctl/ctl/ctl.py")
            # # assert all these subdirectories exist
            # assert fpath.parent.parent.parent.parent.exists(), fpath.parent.parent.parent.parent
            # assert fpath.parent.parent.parent.exists(), fpath.parent.parent.parent
            # assert fpath.parent.parent.exists(), fpath.parent.parent

            print("Launching", name, command_str)
            subprocess.run(
                [
                    "ctl",
                    "job",
                    "run",
                    f"--name={name}",
                    "--shared-host-dir-slow-tolerant",
                    f"--container={job.container}",
                    f"--cpu={job.cpu}",
                    f"--gpu={job.gpu}",
                    "--login",
                    "--wandb",
                    f"--command={command_str}",
                    "--working-dir=/Automatic-Circuit-Discovery",
                    "--shared-host-dir=/home/aconmy/.cache",
                    "--shared-host-dir-mount=/root/.cache",
                    *job.mount_training_options(),
                ],
                check=True,
            )
            print("DONE")
        i += 1

    for (command, process, out, err) in to_wait:
        retcode = process.wait()
        with open(out.name, 'r') as f:
            stdout = f.read()
        with open(err.name, 'r') as f:
            stderr = f.read()

        if retcode != 0 or "nan" in stdout.lower() or "nan" in stderr.lower():
            s = f""" Command {command} exited with code {retcode}.
stdout:
{stdout}
stderr:
{stderr}
"""
            print(s)

import subprocess
import argparse
import os
from pathlib import Path
import shlex
import random

IS_ADRIA = "arthur" not in __file__ and not __file__.startswith("/root")
print("is adria:", IS_ADRIA)

TASKS = ["docstring"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}

def main(
    alg: str, 
    task: str, 
    job: KubernetesJob, 
    testing: bool = False, 
    mod_idx=0,
    num_processes=1,
):

    OUT_RELPATH = Path("arthur_cache")
    OUT_DIR = Path("/root") / OUT_RELPATH

    seed = 1233778640
    random.seed(seed)

    commands = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for metric in METRICS_FOR_TASK[task]:
                if alg == "canonical" and (task == "induction" or metric == "kl_div"):
                    continue

                command = [
                    "python",
                    "notebooks/roc_plot_generator.py",
                    f"--task={task}",
                    f"--reset-network={reset_network}",
                    f"--metric={metric}",
                    f"--alg={alg}",
                    f"--device={'cpu' if testing or not job.gpu else 'cuda'}",
                    # f"--first-cache-cpu=False",
                    f"--torch-num-threads={job.cpu}",
                    f"--out-dir={OUT_DIR}",
                    f"--seed={random.randint(0, 2**31-1)}",
                ]
                if zero_ablation:
                    command.append("--zero-ablation")

                if alg == "acdc" and task == "greaterthan" and metric == "kl_div" and not zero_ablation and not reset_network:
                    command.append("--ignore-missing-score")
                commands.append(command)

    if ON_HOFVARPNIR:
        launch(
            commands,
            name="collect_data",
            job=job,
            synchronous=True,
            just_print_commands=False,
            check_wandb=WandbIdentifier(f"agarriga-col-{alg}-{task[-5:]}-{{i:04d}}b", "collect", "acdc"),
        )

    else:
        for i, command in enumerate(commands):
            subprocess.run(command, check=True)

tasks_for = {
    "acdc": TASKS,
    "16h": TASKS,
    "sp": TASKS,
    "canonical": TASKS,
}

if __name__ == "__main__":
    for alg in ["canonical"]:
        tasks_list = tasks_for[alg]
        for task_idx in range(0, len(tasks_for[alg]), 1): # change to 2 at end to parallelize...
            task = tasks_list[task_idx]
            main(
                alg,
                task,
                KubernetesJob(
                    container="ghcr.io/arthurconmy/automatic-circuit-discovery:tag",
                    cpu=4,
                    gpu=0 if task.startswith("tracr") or alg not in ["acdc", "canonical"] else 1,
                    mount_training=False,
                ),
                testing=False,
            )
