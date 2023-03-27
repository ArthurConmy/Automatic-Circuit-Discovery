import subprocess
import numpy as np 

START = 0.1
STOP = 1.5

for it in range(3, int(1e6)):
    curspace = np.linspace(start=START, stop=STOP, num=it)
    for threshold in curspace[1:-1]:
        subprocess.run(["python", "/mnt/ssd-0/arthurworkspace/TransformerLens/transformer_lens/acdc.py", "--threshold", str(threshold)])