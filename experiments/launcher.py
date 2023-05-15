from pathlib import Path
import subprocess
from typing import Optional, TextIO, List, Tuple
import numpy as np
import shlex
import dataclasses

@dataclasses.dataclass(frozen=True)
class KubernetesJob:
    container: str
    cpu: int
    gpu: int

def launch(commands: List[List[str]], name: str, job: Optional[KubernetesJob] = None):
    to_wait: List[Tuple[str, subprocess.Popen, TextIO, TextIO]] = []

    for i, command in enumerate(commands):
        command_str = shlex.join(command)
        print("Launching", command_str)
        if job is None:
            out = subprocess.call(command)
            assert out == 0
            # base_path = Path(f"/tmp/{name}")
            # base_path.mkdir(parents=True, exist_ok=True)
            # stdout = open(base_path / f"stdout_{i:03d}.txt", "w")
            # stderr = open(base_path / f"stderr_{i:03d}.txt", "w")
            # out = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            # to_wait.append((command_str, out, stdout, stderr))
        else:
            subprocess.run(
                [
                    "ctl",
                    "job",
                    "run",
                    f"--name=agarriga-{name}-{i:03d}",
                    "--shared-host-dir-slow-tolerant",
                    f"--container={job.container}",
                    f"--cpu={job.cpu}",
                    f"--gpu={job.gpu}",
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
            raise RuntimeError(s)
