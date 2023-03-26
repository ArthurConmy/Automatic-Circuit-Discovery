import subprocess
import numpy as np 

START = 0.1
STOP = 1.5

for it in range(2, int(1e6)):
    curspace = np.linspace(start=START, stop=STOP, num=it)
    for threshold in curspace:
        subprocess.run(["python", "acdc.py", "--threshold", str(threshold)])