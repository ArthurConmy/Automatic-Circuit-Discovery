#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import pandas as pd
from pathlib import Path
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(
    usage="Generate AUC tables from CSV files. Pass the data.csv file as an argument fname, e.g python notebooks/auc_tables.py --fname=experiments/results/plots/data.csv"
)
parser.add_argument('--fname', type=str, default="../experiments/results/plots/data.csv")
    
if ipython is None:
    args = parser.parse_args()
else: # make parsing arguments work in jupyter notebook
    args = parser.parse_args(args=[])

fname = Path(args.fname)
data = pd.read_csv(fname)

#%%

# GPT-4 function to make latex AUC table

import pandas as pd
from tabulate import tabulate

def csv_to_latex_table(
    csv_filename, 
    not_metric='kl_div', 
    metric=None,
    weights_type='trained', 
    ablation_type='random_ablation', 
    plot_type='roc_edges'
):
    if (metric is None) == (not_metric is None):
        raise Exception("Exactly one of 'metric' and 'not_metric' must be specified.")

    # Read the CSV data
    data = pd.read_csv(csv_filename)

    # Filter the data
    if metric is None:
        data = data[(data['metric'] != not_metric) & 
                    (data['weights_type'] == weights_type) & 
                    (data['ablation_type'] == ablation_type) &  
                    (data['plot_type'] == plot_type)
                ]
    else:
        data = data[(data['metric'] == metric) &
                    (data['weights_type'] == weights_type) &
                    (data['ablation_type'] == ablation_type) &
                    (data['plot_type'] == plot_type)
                ]   

    # Group by 'task' and 'method' and compute the difference between the max and min 'AUC' within each group
    auc_diffs = data.groupby(['task', 'method'])['auc'].apply(lambda x: x.max() - x.min())

    # Check if there are any groups with a difference greater than 1e-5
    if any(auc_diffs > 1e-5):
        # Print out all 'AUC' entries for these groups
        print("The following task-method combinations have 'AUC' entries with a difference greater than 1e-5:")
        for index, diff in auc_diffs[auc_diffs > 1e-5].iteritems():
            task, method = index
            auc_values = data[(data['task'] == task) & (data['method'] == method)]['auc']
            print(f"Task: {task}, Method: {method}, AUC Values: {auc_values.tolist()}")
        raise ValueError("The 'AUC' entries for each task-method combination should be the same (within a tolerance of 1e-5).")

    # Since the 'AUC' values within each task-method combination are the same, we can simply take the first 'AUC' value in each group
    data = data.groupby(['task', 'method']).first().reset_index()

    # Create a pivot table with 'task' as index, 'method' as columns, and 'AUC' as values
    pivot_table = data.pivot_table(index='task', columns='method', values='auc')

    # Convert the pivot table to LaTeX table format
    latex_table = tabulate(pivot_table, headers='keys', tablefmt='latex', floatfmt=".3f", missingval="N/A")

    return latex_table

# print(csv_to_latex_table('your_file.csv'))  # Replace 'your_file.csv' with your actual CSV file path
print(csv_to_latex_table(fname, ablation_type="random_ablation", weights_type="trained"))

# %%

# Reset networks
for weights_type in ["reset", "trained"]:
    reset_data = data[data["weights_type"] == weights_type]

    # is_kl_div = True

    # if is_kl_div:
    #     df = reset_data[reset_data["metric"] == "kl_div"]
    # else:
    #     df = reset_data[(reset_data["metric"] != "kl_div") | (reset_data["task"] == "tracr")]

    df = reset_data
    df = df.replace({"metric": df.metric.map(lambda x: "other" if x != "kl_div" else x)})

    df2 = df.drop_duplicates(subset=["task", "method", "metric", "ablation_type"])
    print(len(df2), len(df))

    df = df2

    texts = {}


    from math import log10, floor
    def round_to_3(x):
        return round(x, -int(floor(log10(abs(x)))))


    def process_row(row):
        if row["metric"] == "kl_div":
            key = "test_kl_div"
        else:
            key = "test_loss"

        key = "auc"

        i = len(texts)
        distance = max(abs(row[key] - row[key+"_min"]), abs(row[key] - row[key+"_max"]))
        texts[i] =  f"${row[key]:.2g}$ ($\pm{distance:.2g}$)"
        return i

    df["text"] = df.apply(process_row, axis=1)
    out = df.drop("Unnamed: 0", axis=1).pivot_table(index=["task", "metric"],
                                            columns=["ablation_type", "method"],
                                            values="text")

    out = out.applymap(lambda x: texts[x] if isinstance(x, int) else x)

    # %% Export as latex

    out.random_ablation.to_latex(f"{weights_type}_random_ablation.tex", escape=False)
    out.zero_ablation.to_latex(f"{weights_type}_zero_ablation.tex", escape=False)
