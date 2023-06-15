import os
from pathlib import Path
from experiments.launcher import KubernetesJob, launch
import shlex

TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main():
    OUT_DIR = Path(__file__).resolve().parent

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
        possible_files = {"generate_makefile.py", "Makefile"}

        for alg in ["16h", "sp", "acdc", "canonical"]:
            for reset_network in [0, 1]:
                for zero_ablation in [0, 1]:
                    for task in TASKS:
                        for metric in METRICS_FOR_TASK[task]:
                            if alg == "canonical" and metric == "kl_div":
                                # No need to repeat the canonical calculations for both train metrics
                                # (they're the same, nothing is trained)
                                continue

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
        f.write("16h: " + " ".join(sorted(sixteenh_files)) + "\n\n")
        f.write("sp: " + " ".join(sorted(sp_files)) + "\n\n")
        f.write("acdc: " + " ".join(sorted(acdc_files)) + "\n\n")
        f.write("canonical: " + " ".join(sorted(canonical_files)) + "\n\n")
        f.write("trained: " + " ".join(sorted(trained_files)) + "\n\n")
        f.write("reset: " + " ".join(sorted(reset_files)) + "\n\n")
        f.write("zero: " + " ".join(sorted(zero_files)) + "\n\n")
        f.write("random: " + " ".join(sorted(random_files)) + "\n\n")
        f.write("ioi: " + " ".join(sorted(ioi_files)) + "\n\n")
        f.write("docstring: " + " ".join(sorted(docstring_files)) + "\n\n")
        f.write("greaterthan: " + " ".join(sorted(greaterthan_files)) + "\n\n")
        f.write("tracr-reverse: " + " ".join(sorted(tracr_reverse_files)) + "\n\n")
        f.write("tracr-proportion: " + " ".join(sorted(tracr_proportion_files)) + "\n\n")
        f.write("induction: " + " ".join(sorted(induction_files)) + "\n\n")

    print(actual_files - possible_files)
    assert len(actual_files - possible_files) == 0, "There are files that shouldn't be there"

    missing_files = possible_files - actual_files
    print(f"Missing {len(missing_files)} files:")
    for missing_file in missing_files:
        print(missing_file)

if __name__ == "__main__":
    main()
