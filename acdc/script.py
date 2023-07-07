import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", "ioi", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_no_split", "--using-wandb", "--dont-split-qkv", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "logit_diff", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)


if __name__ == '__main__':

    num_gpus = 1 # specify the number of GPUs available
    num_jobs_per_gpu = 2 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = np.linspace(0.02, 0.07, it) # for IOI
        
        correct_val = 0.0575
        factors = [2.0, 1.5, 3.0, 1.3, 1.1, 1.75]
        curspace = [ f / correct_val for f in factors ]
        # curspace = [              
        #     1,
        #     0.63096,
        #     0.39811,
        #     0.25119,
        #     0.15849,
        #     0.1,
        #     0.0631,
        #     0.03981,
        #     0.02512,
        #     0.01585,
        #     0.01,
        #     0.00631,
        #     0.00398,
        #     0.00251,
        #     0.00158,
        #     0.001,
        #     0.00063,
        #     0.0004,
        #     0.00025,
        #     0.00016,
        #     0.0001,
        # ]

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