import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd
# from experiments.launch_spreadsheet import get_all_commands
# from experiments.launch_all_sixteen_heads import get_16h_commands

used = set()
commands = ['make sp-tracr-proportion-l2-False-1.json', 'make sp-tracr-proportion-l2-False-0.json', 'make sp-tracr-proportion-l2-True-1.json', 'make sp-tracr-reverse-l2-True-0.json', 'make sp-tracr-reverse-l2-True-1.json', 'make sp-tracr-reverse-l2-False-1.json', 'make sp-tracr-proportion-l2-True-0.json', 'make sp-tracr-reverse-l2-False-0.json']
# np.random.shuffle(commands)

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    # print(os.getcwd()) # correct
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(commands[threshold].split(" "), env=env)

# subprocess.run(["python", "main.py", "--task", "ioi", "--metric", "kl_div", "--wandb-run-name", str(threshold), "--wandb-project-name", "acdc", "--wandb-group-name", "ioi-zero-redo", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)
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

    num_gpus = 8 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = list(range(len(commands)))

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