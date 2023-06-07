import os
import subprocess
from pathlib import Path
import shlex
import argparse

IS_ADRIA = "arthur" not in __file__ and not __file__.startswith("/root")
print("is adria:", IS_ADRIA)

if IS_ADRIA:
    from experiments.launcher import KubernetesJob, launch

TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main(return_files = False):
    OUT_DIR = Path(__file__).resolve().parent.parent / "acdc" / "media" / "plots_data"

    actual_files = set(os.listdir(OUT_DIR))
    trained_files = []
    reset_files = []
    sixteenh_files = []
    sp_files = []
    acdc_files = []
    canonical_files = []
    random_files = []
    zero_files = []
    ioi_files = []
    docstring_files = []
    greaterthan_files = []
    tracr_reverse_files = []
    tracr_proportion_files = []
    induction_files = []

    with open(OUT_DIR/ "Makefile", "w") as f:
        possible_files = {"analysis_of_rocs.py", "Makefile"}

        for alg in ["16h", "sp", "acdc", "canonical"]:
            for reset_network in [0, 1]:
                for zero_ablation in [0, 1]:
                    for task in TASKS:
                        if alg == "canonical" and task == "induction":
                            continue  # No canonical circuit for induction

                        for metric in METRICS_FOR_TASK[task]:
                            fname = f"{alg}-{task}-{metric}-{bool(zero_ablation)}-{reset_network}.json"
                            possible_files.add(fname)


                            command = [
                                "python",
                                "../../../notebooks/roc_plot_generator.py",
                                f"--task={task}",
                                f"--reset-network={reset_network}",
                                f"--metric={metric}",
                                f"--alg={alg}",
                            ]
                            if zero_ablation:
                                command.append("--zero-ablation")

                            f.write(fname + ":\n" + "\t" + shlex.join(command) + "\n\n")

                            if alg == "16h":
                                sixteenh_files.append(fname)
                            elif alg == "sp":
                                sp_files.append(fname)
                            elif alg == "acdc":
                                acdc_files.append(fname)
                            elif alg == "canonical":
                                canonical_files.append(fname)

                            if reset_network:
                                reset_files.append(fname)
                            else:
                                trained_files.append(fname)

                            if zero_ablation:
                                zero_files.append(fname)
                            else:
                                random_files.append(fname)

                            if task == "ioi":
                                ioi_files.append(fname)
                            elif task == "docstring":
                                docstring_files.append(fname)
                            elif task == "greaterthan":
                                greaterthan_files.append(fname)
                            elif task == "tracr-reverse":
                                tracr_reverse_files.append(fname)
                            elif task == "tracr-proportion":
                                tracr_proportion_files.append(fname)
                            elif task == "induction":
                                induction_files.append(fname)

        f.write("all: " + " ".join(sorted(possible_files)) + "\n\n")
        f.write("16h: " + " ".join(sixteenh_files) + "\n\n")
        f.write("sp: " + " ".join(sp_files) + "\n\n")
        f.write("acdc: " + " ".join(acdc_files) + "\n\n")
        f.write("canonical: " + " ".join(canonical_files) + "\n\n")
        f.write("trained: " + " ".join(trained_files) + "\n\n")
        f.write("reset: " + " ".join(reset_files) + "\n\n")
        f.write("zero: " + " ".join(zero_files) + "\n\n")
        f.write("random: " + " ".join(random_files) + "\n\n")
        f.write("ioi: " + " ".join(ioi_files) + "\n\n")
        f.write("docstring: " + " ".join(docstring_files) + "\n\n")
        f.write("greaterthan: " + " ".join(greaterthan_files) + "\n\n")
        f.write("tracr-reverse: " + " ".join(tracr_reverse_files) + "\n\n")
        f.write("tracr-proportion: " + " ".join(tracr_proportion_files) + "\n\n")
        f.write("induction: " + " ".join(induction_files) + "\n\n")

    print(actual_files - possible_files)
    assert len(actual_files - possible_files) == 0, "There are files that shouldn't be there"

    missing_files = possible_files - actual_files
    print(f"Missing {len(missing_files)} files:")
    for missing_file in missing_files:
        print(missing_file)

    if return_files:
        return actual_files, possible_files



def start_jobs(actual_files, possible_files, just_tracr=False):
    print(actual_files - possible_files)
    assert len(actual_files - possible_files) == 0, "There are files that shouldn't be there"

    missing_files = possible_files - actual_files
    print(f"Missing {len(missing_files)} files:")
    for missing_file in missing_files:
        print(missing_file)

    if just_tracr: 
        missing_files = [f for f in missing_files if "tracr" in f]
        print("\n... but removed some files due to just making tracr files...\n")
        for missing_file in missing_files:
            print(missing_file)

    for name in missing_files:
        name = name.rstrip(".json")
        name = name.split("-")
        reset_network = int(name[-1])
        zero_ablation = bool(name[-2])
        metric = name[-3]
        task = name[1]
        if task == "tracr":
            task = f"tracr-{name[2]}"
        alg = name[0]
        command = [
            "python",
            "notebooks/roc_plot_generator.py",
            f"--task={task}",
            f"--reset-network={reset_network}",
            f"--metric={metric}",
            f"--alg={alg}",
        ]
        if zero_ablation:
            command.append("--zero-ablation")
        try:
            if IS_ADRIA:
                launch([command], name="plots", job=None, synchronous=True)
            else:
                subprocess.run(command, check=True)
        except Exception as e:
            print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-jobs", action="store_true")
    parser.add_argument("--just-tracr", action="store_true")
    RUN_JOBS = parser.parse_args().start_jobs
    JUST_TRACR = parser.parse_args().just_tracr

    output = main(return_files=RUN_JOBS)

    if RUN_JOBS: 
        start_jobs(*output, just_tracr=JUST_TRACR)
