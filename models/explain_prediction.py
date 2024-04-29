"""
Explains the predictions of each model for a single transaction

Example usage:
    $ poetry run python -m models.explain_predictions --tid 69420
"""

import argparse
import json
import math
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--tid",
    help="ID of transaction to explain",
    type=int,
    required=True,
)
args = parser.parse_args()

model_names: tuple[str, ...] = (
    "dist_to_dst_clust_median",
    "dist_to_src_clust_median",
    "isolation_forest",
    "local_outlier_factor",
)

print(
    f"""---------------------------------------{"-"*len(str(args.tid))}
-- Explanation of transaction tid=[{args.tid}] --
---------------------------------------{"-"*len(str(args.tid))}"""
)
transact_df = (
    pd.read_csv("data/input/simdata.csv").drop("is_fraud", axis=1).sample(frac=1.0)
)
transact_row = transact_df.query("tid == @args.tid")
if len(transact_row) == 0:
    print(f"No transaction with tid='{args.tid}' found")
    exit()
print(transact_row.to_string(index=False), "\n")

print("--Anomaly rank under each model--")
for model_name in model_names:
    pred_df = pd.read_csv(f"models/predictions/{model_name}.csv")
    pred_row = pred_df.query("tid==@args.tid")
    anomaly_score_rank = int(round(pred_row["anomaly_score_rank"].iloc[0]))
    print(
        f"    {model_name}: "
        f"{anomaly_score_rank:,}/{pred_df.shape[0]:,} "
        f"(top {math.ceil(100*anomaly_score_rank/pred_df.shape[0])}% most anomalous)"
    )

print("\n--In context of other transactions from the same payment source--")
src: str = transact_row["src"].item()
transact_this_src_df = transact_df.query("src==@src")
transact_this_src_df.insert(
    0, " ", [" --> " if tid == args.tid else "" for tid in transact_this_src_df.tid]
)
print(transact_this_src_df.to_string(index=False))

MAX_BAR_NCHAR: int = 100

for model_name in model_names:
    model_explanation_path: str = f"models/explanations/{model_name}.json"
    if not os.path.exists(model_explanation_path):
        continue
    print(f"\n--Explanation of prediction by model [{model_name}]--")
    print("(Variable contributions to increased anomalousness)")
    with open(model_explanation_path, "r", encoding="utf-8") as file:
        tid_expl = json.load(file)[str(args.tid)]
    tid_expl = dict(sorted(tid_expl.items(), key=lambda tup: -tup[1]))
    biggest_vbl_val = next(iter(tid_expl.values()))
    longest_name_nchar = max([len(name) for name in tid_expl.keys()])
    for vbl_name, vbl_val in tid_expl.items():
        if vbl_val / biggest_vbl_val > 0.05:
            print(
                f"{vbl_name:<{longest_name_nchar}}",
                "|" * int(100 * vbl_val / biggest_vbl_val),
            )
