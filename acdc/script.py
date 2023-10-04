import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", "greaterthan", "--metric", "greaterthan", "--wandb-group-name", "greaterthanabstest", "--wandb-run-name", "greaterthanabstest_"+str(threshold), "--wandb-project-name", "rerun_start", "--using-wandb", "--abs-value-threshold","--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "logit_diff", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
    # subprocess.run(["python", "main.py", "--task", "induction", "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_ioi_abs", "--abs-value-threshold", "--using-wandb", 

if __name__ == '__main__':

    num_gpus = 8 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):        
        curspace = list(
            set([1,
                    0.6309573444801942,
                    0.39810717055349776,
                    0.25118864315095824,
                    0.15848931924611143,
                    0.1,
                    0.06309573444801943,
                    0.039810717055349776,
                    0.025118864315095826,
                    0.01584893192461114,
                    0.01,
                    0.006309573444801936,
                    0.003981071705534973,
                    0.002511886431509582,
                    0.001584893192461114,
                    0.001,
                    0.0006309573444801936,
                    0.00039810717055349735,
                    0.0002511886431509582,
                    0.00015848931924611142,
                    0.0001,
                    6.309573444801929e-05,
                    3.9810717055349695e-05,
                    2.5118864315095825e-05,
                    1.584893192461114e-05,
                    1e-05,
                    6.30957344480193e-06,
                    3.981071705534969e-06,
                    2.5118864315095823e-06,
                    1.584893192461114e-06,
                    1e-06,
                ]
            ) - set(
                [
                    1,
                    0.39811,
                    0.15849,
                    0.1,
                    0.0631,
                    0.03981,
                    0.00251,
                    0.0004,
                    0.00025,
                    0.00016,
                    0.0001,
                ]
            )
        )

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