from pathlib import Path
import subprocess
from typing import Optional, TextIO, List, Tuple
import numpy as np
import shlex
import dataclasses
import wandb

@dataclasses.dataclass(frozen=True)
class KubernetesJob:
    container: str
    cpu: int
    gpu: int
    mount_training: bool=False

    def mount_training_options(self) -> list[str]:
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
                    "--never-restart",
                    f"--command={command_str}",
                    "--working-dir=/Automatic-Circuit-Discovery",
                    "--shared-host-dir=/home/agarriga/.cache",
                    "--shared-host-dir-mount=/root/.cache",
                    *job.mount_training_options(),
                ],
                check=True,
            )
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
