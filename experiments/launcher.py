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

@dataclasses.dataclass(frozen=True)
class WandbIdentifier:
    run_name: str
    group_name: str
    project: str


def launch(commands: List[List[str]], name: str, job: Optional[KubernetesJob] = None, check_wandb: Optional[WandbIdentifier]=None, ids_for_worker=range(0, 10000000)):
    to_wait: List[Tuple[str, subprocess.Popen, TextIO, TextIO]] = []

    assert len(commands) <= 100_000, "Too many commands for 5 digits"

    existing_names = {'agarriga-acdc-spreadsheet-00209', 'agarriga-acdc-spreadsheet-00203',
                      'agarriga-acdc-spreadsheet-00436', 'agarriga-acdc-spreadsheet-00253', 'agarriga-acdc-spreadsheet-00306',
                      'agarriga-acdc-spreadsheet-00370', 'agarriga-acdc-spreadsheet-00201', 'agarriga-acdc-spreadsheet-00283',
                      'agarriga-acdc-spreadsheet-00441', 'agarriga-acdc-spreadsheet-00206', 'agarriga-acdc-spreadsheet-00104',
                      'agarriga-acdc-spreadsheet-00281', 'agarriga-acdc-spreadsheet-00375', 'agarriga-acdc-spreadsheet-00303',
                      'agarriga-acdc-spreadsheet-00361', 'agarriga-acdc-spreadsheet-00105', 'agarriga-acdc-spreadsheet-00438',
                      'agarriga-acdc-spreadsheet-00358', 'agarriga-acdc-spreadsheet-00288', 'agarriga-acdc-spreadsheet-00372',
                      'agarriga-acdc-spreadsheet-00222', 'agarriga-acdc-spreadsheet-00053', 'agarriga-acdc-spreadsheet-00435',
                      'agarriga-acdc-spreadsheet-00382', 'agarriga-acdc-spreadsheet-00402', 'agarriga-acdc-spreadsheet-00373',
                      'agarriga-acdc-spreadsheet-00371', 'agarriga-acdc-spreadsheet-00204', 'agarriga-acdc-spreadsheet-00252',
                      'agarriga-acdc-spreadsheet-00368', 'agarriga-acdc-spreadsheet-00401', 'agarriga-acdc-spreadsheet-00106',
                      'agarriga-acdc-spreadsheet-00301', 'agarriga-acdc-spreadsheet-00208', 'agarriga-acdc-spreadsheet-00442',
                      'agarriga-acdc-spreadsheet-00223', 'agarriga-acdc-spreadsheet-00403', 'agarriga-acdc-spreadsheet-00214',
                      'agarriga-acdc-spreadsheet-00210', 'agarriga-acdc-spreadsheet-00376', 'agarriga-acdc-spreadsheet-00218',
                      'agarriga-acdc-spreadsheet-00362', 'agarriga-acdc-spreadsheet-00287', 'agarriga-acdc-spreadsheet-00000',
                      'agarriga-acdc-spreadsheet-00221', 'agarriga-acdc-spreadsheet-00215', 'agarriga-acdc-spreadsheet-00050',
                      'agarriga-acdc-spreadsheet-00366', 'agarriga-acdc-spreadsheet-00364', 'agarriga-acdc-spreadsheet-00356',
                      'agarriga-acdc-spreadsheet-00103', 'agarriga-acdc-spreadsheet-00225', 'agarriga-acdc-spreadsheet-00300',
                      'agarriga-acdc-spreadsheet-00440', 'agarriga-acdc-spreadsheet-00360', 'agarriga-acdc-spreadsheet-00304',
                      'agarriga-acdc-spreadsheet-00224', 'agarriga-acdc-spreadsheet-00100', 'agarriga-acdc-spreadsheet-00152',
                      'agarriga-acdc-spreadsheet-00275', 'agarriga-acdc-spreadsheet-00175', 'agarriga-acdc-spreadsheet-00217',
                      'agarriga-acdc-spreadsheet-00381', 'agarriga-acdc-spreadsheet-00251', 'agarriga-acdc-spreadsheet-00211',
                      'agarriga-acdc-spreadsheet-00286', 'agarriga-acdc-spreadsheet-00285', 'agarriga-acdc-spreadsheet-00369',
                      'agarriga-acdc-spreadsheet-00437', 'agarriga-acdc-spreadsheet-00432', 'agarriga-acdc-spreadsheet-00001',
                      'agarriga-acdc-spreadsheet-00216', 'agarriga-acdc-spreadsheet-00444', 'agarriga-acdc-spreadsheet-00282',
                      'agarriga-acdc-spreadsheet-00250', 'agarriga-acdc-spreadsheet-00219', 'agarriga-acdc-spreadsheet-00212',
                      'agarriga-acdc-spreadsheet-00377', 'agarriga-acdc-spreadsheet-00353', 'agarriga-acdc-spreadsheet-00305',
                      'agarriga-acdc-spreadsheet-00220', 'agarriga-acdc-spreadsheet-00351', 'agarriga-acdc-spreadsheet-00359',
                      'agarriga-acdc-spreadsheet-00367', 'agarriga-acdc-spreadsheet-00302', 'agarriga-acdc-spreadsheet-00151',
                      'agarriga-acdc-spreadsheet-00445', 'agarriga-acdc-spreadsheet-00443', 'agarriga-acdc-spreadsheet-00052',
                      'agarriga-acdc-spreadsheet-00075', 'agarriga-acdc-spreadsheet-00102', 'agarriga-acdc-spreadsheet-00378',
                      'agarriga-acdc-spreadsheet-00289', 'agarriga-acdc-spreadsheet-00307', 'agarriga-acdc-spreadsheet-00350',
                      'agarriga-acdc-spreadsheet-00200', 'agarriga-acdc-spreadsheet-00379', 'agarriga-acdc-spreadsheet-00213',
                      'agarriga-acdc-spreadsheet-00207', 'agarriga-acdc-spreadsheet-00354', 'agarriga-acdc-spreadsheet-00205',
                      'agarriga-acdc-spreadsheet-00202', 'agarriga-acdc-spreadsheet-00284', 'agarriga-acdc-spreadsheet-00254',
                      'agarriga-acdc-spreadsheet-00278', 'agarriga-acdc-spreadsheet-00363', 'agarriga-acdc-spreadsheet-00150',
                      'agarriga-acdc-spreadsheet-00355', 'agarriga-acdc-spreadsheet-00279', 'agarriga-acdc-spreadsheet-00365',
                      'agarriga-acdc-spreadsheet-00277', 'agarriga-acdc-spreadsheet-00280', 'agarriga-acdc-spreadsheet-00276',
                      'agarriga-acdc-spreadsheet-00439', 'agarriga-acdc-spreadsheet-00400', 'agarriga-acdc-spreadsheet-00433',
                      'agarriga-acdc-spreadsheet-00051', 'agarriga-acdc-spreadsheet-00357', 'agarriga-acdc-spreadsheet-00352',
                      'agarriga-acdc-spreadsheet-00101', 'agarriga-acdc-spreadsheet-00434', 'agarriga-acdc-spreadsheet-00374',
                      'agarriga-acdc-spreadsheet-00383', 'agarriga-acdc-spreadsheet-00380'}

    print(f"Launching {len(commands)} jobs")
    for i, command in enumerate(commands):
        if i not in ids_for_worker:
            print(f"Skipping {name} because it's not my turn, {i} not in {ids_for_worker}")
            continue

        command_str = shlex.join(command)


        if check_wandb is not None:
            # HACK this is pretty vulnerable to duplicating work if the same run is launched in close succession,
            # it's more to be able to restart
            api = wandb.Api()
            name = check_wandb.run_name.format(i=i)
            if name in existing_names:
                print(f"Skipping {name} because it already exists")
                continue

            # runs = api.runs(path=f"remix_school-of-rock/{check_wandb.project}", filters={"group": check_wandb.group_name})
            # existing_names = existing_names.union({r.name for r in runs})
            # print("Runs that exist: ", existing_names)
            # if name in existing_names:
            #     print(f"Run {name} already exists, skipping")
            #     continue

        print("Launching", name, command_str)
        if job is None:
            # out = subprocess.run(command)
            # assert out.returncode == 0, f"Command return={out.returncode} != 0"
            # Old code for async launching all jobs
            base_path = Path(f"/tmp/{name}")
            base_path.mkdir(parents=True, exist_ok=True)
            stdout = open(base_path / f"stdout_{i:03d}.txt", "w")
            stderr = open(base_path / f"stderr_{i:03d}.txt", "w")
            out = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            to_wait.append((command_str, out, stdout, stderr))
        else:
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
            print(s)
