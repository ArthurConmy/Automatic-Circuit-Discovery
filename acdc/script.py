import subprocess
import numpy as np 
from math import gcd

START = -3
STOP = 1
div = 2

for it in range(3, int(1e6)):
    curspace = np.linspace(0.05, 0.1, it)
    # curspace = np.logspace(-2, -1, it)
    # curspace = [
    #     # 0.075,
    #     # 0.4,
    #     # 0.5,
    #     # 0.3,
    #     # 0.25,
    #     # 0.2,
    #     0.067,
    # ]

    if not isinstance(curspace, list):
        curspace = curspace[1:-1]

    print(curspace)
    for threshold_idx, threshold in list(enumerate(curspace)):
        if not isinstance(curspace, list):
            if gcd(threshold_idx, it) != 1:
                continue
        
        subprocess.run(["python", "main.py", "--wandb-project-name", "arthur_more_docstring", "--task", "docstring", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse"])

    if isinstance(curspace, list):
        break