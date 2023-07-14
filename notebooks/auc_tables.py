#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
import io
import numpy as np

import pandas as pd
from pathlib import Path
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(
    usage="Generate AUC tables from CSV files. Pass the data.csv file as an argument fname, e.g python notebooks/auc_tables.py --fname=experiments/results/plots/data.csv"
)
parser.add_argument('--in-fname', type=str, default="experiments/results/plots/data.csv")
parser.add_argument('--out-fname', type=str, default="experiments/results/auc_tables.tex")
    
if ipython is None:
    args = parser.parse_args()
else: # make parsing arguments work in jupyter notebook
    args = parser.parse_args(args=[])

data = pd.read_csv(args.in_fname)

# %%
with io.StringIO() as buf:
    for key in ["auc"]:  # ["test_kl_div", "test_loss", "auc"]:
        for weights_type in ["trained"]:  # ["reset", "trained"]
            df = data[(data["weights_type"] == weights_type)]
            df = df.replace({"metric": df.metric.map(lambda x: "other" if x != "kl_div" else x)})
            df = df.drop_duplicates(subset=["task", "method", "metric", "ablation_type", "plot_type"])

            def process_metric_pretty(row):
                if row["metric"] == "kl_div":
                    return "KL"
                else:
                    return "Loss"

            df["metric_pretty"] = df.apply(process_metric_pretty, axis=1)
            out = df.drop("Unnamed: 0", axis=1).pivot_table(
                index=["metric_pretty", "task"], columns=["ablation_type", "plot_type", "method"], values=key
            )
            # Needed to handle non-AUC keys
            # out = out.applymap(lambda x: None if x == -1 else (texts[x] if isinstance(x, int) else x))
            out = out.dropna(axis=0)

            # %% Export as latex
            def export_table(out, name):
                def make_bold_column(row):
                    out = pd.Series(dtype=np.float64)
                    for plot_type in ["roc_edges", "roc_nodes"]:
                        the_max = row.loc[plot_type].max()
                        out = pd.concat(
                            [
                                out,
                                pd.Series(
                                    data=[
                                        f"\\textbf{{{x:.3f}}}" if x == the_max else f"{x:.3f}"
                                        for x in row.loc[plot_type]
                                    ],
                                    index=row.loc[[plot_type]].index,
                                ),
                            ]
                        )
                    return out

                old_out = out
                out = out.apply(make_bold_column, axis=1)
                out.columns = pd.MultiIndex.from_tuples(out.columns)
                out.style.to_latex(buf, hrules=True, environment="table", caption=name)

            export_table(out.random_ablation, f"Key={key}, weights_type={weights_type}, Random Ablation")
            export_table(out.zero_ablation, f"Key={key}, weights_type={weights_type}, Zero Ablation")

    with open(args.out_fname, "w") as f:
        f.write(
            r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\begin{document}

WARNING: you need to add some vertical and horizontal rows for this to look good

"""
        )
        f.write(buf.getvalue().replace("_", "\\_"))
        f.write(
            r"""\end{document}
"""
        )
