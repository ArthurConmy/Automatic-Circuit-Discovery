import subprocess
import numpy as np 
from math import gcd

START = 0.0
STOP = 5.0

for it in range(3, int(1e6)):
    curspace = np.linspace(start=START, stop=STOP, num=it)
    for threshold_idx, threshold in list(enumerate(curspace))[1:-1]:
        if gcd(threshold_idx, it) != 1:
            continue
        subprocess.run(["python", "acdc.py", "--using-wandb", "--threshold", str(threshold), "--zero-ablation"])