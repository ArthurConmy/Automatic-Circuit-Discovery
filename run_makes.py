import multiprocessing
import subprocess

def run_make_command(cmd):
    """
    Function to run a shell command.
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"Error executing {cmd}: {err.decode()}")
    else:
        print(f"Output of {cmd}: {out.decode()}")

if __name__ == '__main__':
    # List of make commands to run
    commands = """16h-docstring-docstring_metric-False-0.json 16h-docstring-docstring_metric-True-0.json 16h-docstring-kl_div-False-0.json 16h-docstring-kl_div-True-0.json 16h-induction-nll-False-0.json 16h-induction-nll-True-0.json 16h-tracr-proportion-l2-False-0.json 16h-tracr-proportion-l2-True-0.json 16h-tracr-reverse-l2-False-0.json 16h-tracr-reverse-l2-True-0.json""".split(" ")
    commands = ["make "+command for command in commands]

    # Create a pool of workers
    pool = multiprocessing.Pool(processes=len(commands))  # 6 processes for 6 commands

    # Run the commands in parallel
    pool.map(run_make_command, commands)

    # Close the pool
    pool.close()
    pool.join()