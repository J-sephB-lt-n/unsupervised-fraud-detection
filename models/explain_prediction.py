"""
Explains the predictions of each model for a single transaction

Example usage:
    $ poetry run python -m models.explain_predictions --tid 69420
"""

import argparse
import math

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

transact_df = (
    pd.read_csv("data/input/simdata.csv").drop("is_fraud", axis=1).sample(frac=1.0)
)
transact_row = transact_df.query("tid == @args.tid")
if len(transact_row) == 0:
    print(f"No transaction with tid='{args.tid}' found")
    exit()
src: str = transact_row["src"].item()
transact_this_src_df = transact_df.query("src==@src")
transact_this_src_df.insert(
    0, " ", [" --> " if tid == args.tid else "" for tid in transact_this_src_df.tid]
)
print(
    "Transaction of interest in context of other transactions from the same payment source:"
)
print(transact_this_src_df.to_string(index=False))

print("--Anomaly rank under each model--")
for model_name in model_names:
    pred_df = pd.read_csv(f"models/predictions/{model_name}.csv")
    pred_row = pred_df.query("tid==@args.tid")
    anomaly_score_rank = int(round(pred_row["anomaly_score_rank"].iloc[0]))
    print(
        f"{model_name}: "
        f"{anomaly_score_rank:,}/{pred_df.shape[0]:,} "
        f"(top {math.ceil(100*anomaly_score_rank/pred_df.shape[0])}% most anomalous)"
    )
