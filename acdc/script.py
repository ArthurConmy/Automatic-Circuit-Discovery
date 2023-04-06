import subprocess
import numpy as np 
from math import gcd

START = -3
STOP = 1
div = 2

for it in range(3, int(1e6)):
    curspace = np.logspace(-2, 1, it) / 5
    for threshold_idx, threshold in list(enumerate(curspace))[1:-1]:
        if gcd(threshold_idx, it) != 1:
            continue
        subprocess.run(["python", "main.py", "--wandb-project-name", "arthurinduction_randomabl_reverse", "--task", "induction", "--using-wandb", "--threshold", str(threshold), "--indices-mode", "reverse"])