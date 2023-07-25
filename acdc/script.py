import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", "reproduce_induction_" + str(threshold), "--wandb-project-name", "acdc", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
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

    num_gpus = 2 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = np.linspace(0.02, 0.07, it) # for IOI
        
        curspace = [1,
              0.63096,
              0.39811,
              0.25119,
              0.15849,
              0.1,
              0.0631,
              0.03981,
              0.02512,
              0.01585,
              0.01,
              0.00631,
              0.00398,
              0.00251,
              0.00158,
              0.001,
              0.00063,
              0.0004,
              0.00025,
              0.00016,
              0.0001,
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