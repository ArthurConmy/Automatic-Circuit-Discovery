import pandas as pd
df = pd.read_csv("data.csv")
df
df[(df["method"] == "ACDC") & (df["ablation_type"] == "random_ablation")]
df[(df["method"] == "ACDC") & (df["ablation_type"] == "random_ablation")]
def f(df):
    print(f"{df.auc.mean():.2f} ({df.auc.min():.2f}--{df.auc.max():.2f})")
f(df[(df["method"] == "ACDC") & (df["ablation_type"] == "random_ablation")])
f(df[(df["method"] == "HISP") & (df["ablation_type"] == "random_ablation")])
f(df[(df["method"] == "SP") & (df["ablation_type"] == "random_ablation")])
f(df[(df["method"] == "SP") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "greaterthan")])
f(df[(df["method"] == "ACDC") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "greaterthan")])
f(df[(df["method"] == "HISP") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "greaterthan")])
f(df[(df["method"] == "ACDC") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "tracr-proportion")])
f(df[(df["method"] == "SP") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "tracr-proportion")])
f(df[(df["method"] == "HISP") & (df["ablation_type"] == "random_ablation") & (df["metric"] != "kl_div") & (df["task"] == "tracr-proportion")])
%history
