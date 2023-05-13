import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

START = -3
STOP = 1
div = 2

thresholds = 10 ** np.linspace(-2, 0.5, 21)
used = set()
used.add(0.005)
used.add(0.0016666666666666663)

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", "greaterthan", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_greaterthan_sweep", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)

if __name__ == '__main__':
    num_gpus = 8 # specify the number of GPUs available
    num_jobs_per_gpu = 2 # specify the number of jobs per GPU
    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = 0.005 * np.logspace(-1, 1, it)

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