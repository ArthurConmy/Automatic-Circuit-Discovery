from copy import deepcopy
import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()
the_curspace = list(set([
    0.1,
    0.0631,
    0.0631,
    0.03981,
    0.03981,
    0.02512,
    0.02512,
    0.01585,
    0.01585,
    0.01,
    0.01,
    0.00631,
    0.00631,
    0.00398,
    0.00398,
    0.00251,
    0.00251,
    0.00158,
    0.00158,
    0.001,
    0.001,
    0.00063,
    0.00063,
    0.0004,
    0.0004,
    0.00025,
    0.00025,
    0.00016,
    0.00016,
    0.0001,
    0.0001,
    6e-05,
    6e-05,
    4e-05,
    4e-05,
    3e-05,
    3e-05,
    2e-05,
    2e-05,
    1e-05,
    1e-05,
    6.309573444801943e-06,
    3.981071705534969e-06,
    2.5118864315095823e-06,
    1.584893192461114e-06,
    1e-06,
    6.309573444801942e-07,
    3.981071705534969e-07,
    2.511886431509583e-07,
    1.584893192461114e-07,
    1e-07,
    6.30957344480193e-08,
    3.9810717055349696e-08,
    2.511886431509582e-08,
    1.5848931924611143e-08,
    1e-08,
    6.309573444801943e-09,
    3.981071705534969e-09,
    2.511886431509582e-09,
    1.584893192461111e-09,
    1e-09,
]))

idx = 0

def run_script(task, threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", "main.py", "--task", task, "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_tracr_fix", "--using-wandb", "--metric", "l2", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)

def run_roc_plot(task, alg, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(
        [
            "python",
            "../notebooks/roc_plot_generator.py",
            f"--task={task}",
            "--metric=l2",
            f"--alg={alg}",
        ],
        env=env,
    )

if __name__ == '__main__':

    num_gpus = 8 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    idx = 0
    for task in ["tracr-proportion"]:
        for alg in ["canonical"]:
            jobs.append(pool.apply_async(run_roc_plot, (task, alg, idx%num_gpus)))
            idx+=1

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()

# from copy import deepcopy
# import os
# import subprocess
# import numpy as np 
# import multiprocessing
# from math import gcd

# used = set()
# the_curspace = list(set([
#     0.1,
#     0.0631,
#     0.0631,
#     0.03981,
#     0.03981,
#     0.02512,
#     0.02512,
#     0.01585,
#     0.01585,
#     0.01,
#     0.01,
#     0.00631,
#     0.00631,
#     0.00398,
#     0.00398,
#     0.00251,
#     0.00251,
#     0.00158,
#     0.00158,
#     0.001,
#     0.001,
#     0.00063,
#     0.00063,
#     0.0004,
#     0.0004,
#     0.00025,
#     0.00025,
#     0.00016,
#     0.00016,
#     0.0001,
#     0.0001,
#     6e-05,
#     6e-05,
#     4e-05,
#     4e-05,
#     3e-05,
#     3e-05,
#     2e-05,
#     2e-05,
#     1e-05,
#     1e-05,
#     6.309573444801943e-06,
#     3.981071705534969e-06,
#     2.5118864315095823e-06,
#     1.584893192461114e-06,
#     1e-06,
#     6.309573444801942e-07,
#     3.981071705534969e-07,
#     2.511886431509583e-07,
#     1.584893192461114e-07,
#     1e-07,
#     6.30957344480193e-08,
#     3.9810717055349696e-08,
#     2.511886431509582e-08,
#     1.5848931924611143e-08,
#     1e-08,
#     6.309573444801943e-09,
#     3.981071705534969e-09,
#     2.511886431509582e-09,
#     1.584893192461111e-09,
#     1e-09,
# ]))

# idx = 0

# def run_script(task, threshold, gpu_id):
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     subprocess.run(["python", "main.py", "--task", task, "--wandb-run-name", str(threshold), "--wandb-project-name", "arthur_tracr_fix", "--using-wandb", "--metric", "l2", "--threshold", str(threshold), "--indices-mode", "reverse", "--first-cache-cpu", "False", "--second-cache-cpu", "False"], env=env)

# def run_roc_plot(task, alg):
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     subprocess.run(
#         [
#             "python",
#             "../notebooks/roc_plot_generator.py",
#             f"--task={task}"
#             f"--alg={alg}",
#         ]
#     )

# if __name__ == '__main__':

#     num_gpus = 8 # specify the number of GPUs available
#     num_jobs_per_gpu = 4 # specify the number of jobs per GPU

#     pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
#     jobs = []

#     for task in ["tracr-proportion", "tracr-reverse"]:
#         for it in range(3, int(1e6)):
#             curspace = deepcopy(the_curspace)

#             if not isinstance(curspace, list):
#                 curspace = curspace[1:-1]

#             for threshold_idx, threshold in list(enumerate(curspace)):
#                 if threshold in used:
#                     continue
#                 used.add(threshold)

#                 gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
#                 idx+=1
#                 # if idx>33:
#                 jobs.append(pool.apply_async(run_script, (task, threshold, gpu_id)))

#             if isinstance(curspace, list):
#                 break

#     # wait for all jobs to finish
#     for job in jobs:
#         job.get()

#     # clean up
#     pool.close()
#     pool.join()