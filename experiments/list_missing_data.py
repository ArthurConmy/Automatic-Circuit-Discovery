import os
from pathlib import Path
from experiments.launcher import KubernetesJob, launch
import shlex

TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main():
    OUT_DIR = Path(__file__).resolve().parent.parent / "acdc" / "media" / "plots_data"

    actual_files = set(os.listdir(OUT_DIR))
    trained_files = []
    reset_files = []
    sixteenh_files = []
    sp_files = []
    acdc_files = []
    random_files = []
    zero_files = []
    ioi_files = []
    docstring_files = []
    greaterthan_files = []
    tracr_reverse_files = []
    tracr_proportion_files = []
    induction_files = []

    with open(OUT_DIR/ "Makefile", "w") as f:
        possible_files = {"analysis_of_rocs.py"}

        for alg in ["16h", "sp", "acdc"]:
            for reset_network in [0, 1]:
                for zero_ablation in [0, 1]:
                    for task in TASKS:
                        for metric in METRICS_FOR_TASK[task]:
                            fname = f"{alg}-{task}-{metric}-{bool(zero_ablation)}-{reset_network}.json"
                            possible_files.add(fname)


                            command = [
                                "python",
                                "/Users/adria/Documents/2023/ACDC/Automatic-Circuit-Discovery/notebooks/roc_plot_generator.py",
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

        f.write("all: " + " ".join(possible_files) + "\n\n")
        f.write("16h: " + " ".join(sixteenh_files) + "\n\n")
        f.write("sp: " + " ".join(sp_files) + "\n\n")
        f.write("acdc: " + " ".join(acdc_files) + "\n\n")
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


def start_jobs(actual_files, possible_files):
    print(actual_files - possible_files)
    assert len(actual_files - possible_files) == 0, "There are files that shouldn't be there"

    missing_files = possible_files - actual_files
    print(f"Missing {len(missing_files)} files:")
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
            launch([command], name="plots", job=None, synchronous=True)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
