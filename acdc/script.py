import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id, zero_ablation, metric, absthreshold):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cli_args = ["python", "main.py", "--task", "docstring", "--wandb-group-name", "docstringmassive", "--wandb-run-name", str(threshold), "--wandb-project-name", "rerun_start", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False", "--metric", metric]
    if zero_ablation: cli_args.append("--zero-ablation")
    if absthreshold: cli_args.append("--abs-value-threshold")
    subprocess.run(cli_args, env=env)
    # subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "logit_diff", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", 

if __name__ == '__main__':

    num_gpus = 1 # specify the number of GPUs available
    num_jobs_per_gpu = 2 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):        
        curspace = [
            3.16228,
            2.37137,
            1.77828,
            1.33352,
            1,
            0.74989,
            0.56234,
            0.4217,
            0.31623,
            0.23714,
            0.13335,
            0.1,
            0.05623,
            0.03162,
            0.02371,
            0.01778,
            0.01334,
            0.01,
            0.00630957344480193,
            0.003981071705534973,
            0.002511886431509582,
            0.001584893192461114,
            0.001,
            0.000630957344480193,
            0.00039810717055349735,
            0.00025118864315095795,
            0.00015848931924611142,
            0.0001,
        ]
        real_curspace = []
        for zero_ablation in [False, True]:
            for metric in ["kl_div", "docstring_metric"]:
                for absthreshold in [False, True]:
                    for threshold in curspace:
                        real_curspace.append([threshold, zero_ablation, metric, absthreshold])
        curspace = real_curspace

        if not isinstance(curspace, list):
            curspace = curspace[1:-1]

        for threshold_idx, threshold in list(enumerate(curspace)):
            if threshold[0] in used:
                continue
            used.add(threshold[0])

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold[0], gpu_id, threshold[1], threshold[2], threshold[3])))

        if isinstance(curspace, list):
            break

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()