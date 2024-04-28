"""
Unsupervised anomaly detection using distance to cluster median within
each payment destination
"""

from typing import Final

import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler


TRAIN_DATA_PATH: Final[str] = "feature_eng/output/train_data.csv"
PRED_OUTPUT_PATH: Final[str] = "models/predictions/dist_to_dst_clust_median.csv"

data: pd.DataFrame = pd.read_csv(TRAIN_DATA_PATH)
train_X: pd.DataFrame = data[
    [
        "tid",
        "dst",
        "amt",
        "dst_ratio_to_min_amt",
        "dst_ratio_to_mean_amt",
        "dst_ratio_to_median_amt",
        "dst_ratio_to_max_amt",
        "dst_n_transactions_this_day_of_week",
        "dst_prop_transactions_this_day_of_week",
        "dst_n_transactions_this_time",
        "dst_prop_transactions_this_time",
        "dst_n_transactions_this_src",
        "dst_prop_transactions_this_src",
    ]
]
for col in (
    "dst_n_transactions_this_day_of_week",
    "dst_n_transactions_this_time",
    "dst_n_transactions_this_src",
):
    train_X = train_X.astype({col: "float64"})
# train_y = data[["is_fraud"]]

standard_scaler = StandardScaler()
cols_to_scale: list[str] = [col for col in train_X.columns if col not in ["tid", "dst"]]
train_X.loc[:, cols_to_scale] = standard_scaler.fit_transform(train_X[cols_to_scale])

unique_dst: list[str] = train_X["dst"].drop_duplicates().tolist()
train_X = train_X.set_index(["dst", "tid"])

rows_list = []
for dst in tqdm.tqdm(unique_dst):
    dfrows = train_X.query("dst==@dst")
    vectors = dfrows.values
    median_row = np.median(vectors, axis=0)
    rows_list.append(
        pd.DataFrame(
            {"anomaly_score": [np.linalg.norm(v - median_row) for v in vectors]},
            index=pd.MultiIndex.from_tuples(dfrows.index, names=["dst", "tid"]),
        )
    )

train_pred = pd.concat(rows_list, axis=0)
assert train_pred.shape[0] == train_X.shape[0]
train_pred["anomaly_score_rank"] = train_pred["anomaly_score"].rank(ascending=False)
train_pred["model"] = "dist_to_dst_clust_median"
train_pred = train_pred.reset_index(level=["tid"])
train_pred = train_pred[["tid", "model", "anomaly_score", "anomaly_score_rank"]]
train_pred.to_csv(PRED_OUTPUT_PATH, index=False)
print(f"predictions written to '{PRED_OUTPUT_PATH}'")
