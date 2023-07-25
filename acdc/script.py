import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id+1)
    # subprocess.run(["python", "main.py", "--task", "ioi", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_no_split", "--using-wandb", "--dont-split-qkv", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "logit_diff", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    subprocess.run(["python", "subnetwork_probing/train.py", "--task", "docstring", "--lambda-reg", str(threshold)] + """--zero-ablation=0
--wandb-name=my_edge_runs
--epochs=10000
--wandb-project=edgesp
--lr=0.001
--wandb-entity=remix_school-of-rock
--wandb-mode=online
--loss-type=docstring_metric
--sp=edge""".split('\n'), env=env)


if __name__ == '__main__':

    num_gpus = 3 # specify the number of GPUs available
    num_jobs_per_gpu = 2 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = np.linspace(0.02, 0.07, it) # for IOI
        
        curspace = [250,
            230,
            210,
            190,
            170,
            150,
            130,
            110,
            90,
            70,
            50,
            30,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0.631,
            0.398,
            0.251,
            0.158,
            0.1,
            0.063,
            0.04,
            0.025,
            0.016,
            0.01,
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