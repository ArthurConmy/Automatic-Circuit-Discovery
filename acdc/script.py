import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", "tracr-reverse", "--wandb-run-name", "tracr_new_" + str(threshold), "--metric", "l2", "--wandb-project-name", "acdc", "--wandb-group-name", "acdc-tracr-neurips-6", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)

    # subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "logit_diff", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
#     subprocess.run(["python", "subnetwork_probing/train.py", "--task", "docstring", "--lambda-reg", str(threshold)] + """--zero-ablation=0
# --wandb-name=my_edge_runs
# --wandb-project=edgesp
# --lr=0.001
# --wandb-entity=remix_school-of-rock
# --wandb-mode=online
# --loss-type=docstring_metric
# --sp=edge""".split('\n'), env=env)


if __name__ == '__main__':

    num_gpus = 1 # specify the number of GPUs available
    num_jobs_per_gpu = 8 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = np.linspace(0.02, 0.07, it) # for IOI
        curspace = [0.1,
            0.0631,
            0.02512,
            0.01585,
            0.01,
            0.00631,
            0.00398,
            0.00158,
            0.001,
            0.00025,
            0.00016,
            0.0001,
            6e-05,
            4e-05,
            3e-05,
            1e-05,
            6.309573444801943e-06,
            3.981071705534969e-06,
            1.584893192461114e-06,
            6.309573444801942e-07,
            3.981071705534969e-07,
            1.584893192461114e-07,
            1e-07,
            6.30957344480193e-08,
            3.9810717055349696e-08,
            2.511886431509582e-08,
            1e-08,
            6.309573444801943e-09,
            3.981071705534969e-09,
            2.511886431509582e-09,
            1.584893192461111e-09,
            1e-09,
        ]

        if not isinstance(curspace, list):
            curspace = curspace[1:-1]

        for threshold_idx, threshold in list(enumerate(curspace)):
            if threshold in used:
                continue
            used.add(threshold)

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold, gpu_id)))

        if isinstance(curspace, list):
            break

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()