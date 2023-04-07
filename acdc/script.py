import subprocess
import numpy as np 
from math import gcd

START = -3
STOP = 1
div = 2

for it in range(3, int(1e6)):
    # curspace = np.linspace(0.2, 0.1, it)
    curspace = np.logspace(-2, -1, it)
    print(curspace)
    for threshold_idx, threshold in list(enumerate(curspace))[1:-1]:
        if gcd(threshold_idx, it) != 1:
            continue
        subprocess.run(["python", "main.py", "--wandb-project-name", "arthur_more_docstring", "--task", "docstring", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse"])