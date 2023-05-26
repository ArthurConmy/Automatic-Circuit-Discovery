import pandas as pd
from pathlib import Path

data = pd.read_csv("acdc/media/plots/data.csv")

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
        i = len(texts)
        distance = max(abs(row[key] - row[key+"_min"]), abs(row[key] - row[key+"_max"]))
        texts[i] =  f"${row[key]:.2g}$ ($\pm{distance:.2g}$)"
        return i

    df["text"] = df.apply(process_row, axis=1)
    # %%
    out = df.drop("Unnamed: 0", axis=1).pivot_table(index=["task", "metric"],
                                            columns=["ablation_type", "method"],
                                            values="text")

    out = out.applymap(lambda x: texts[x] if isinstance(x, int) else x)

    # %% Export as latex

    out.random_ablation.to_latex(f"{weights_type}_random_ablation.tex", escape=False)
    out.zero_ablation.to_latex(f"{weights_type}_zero_ablation.tex", escape=False)
