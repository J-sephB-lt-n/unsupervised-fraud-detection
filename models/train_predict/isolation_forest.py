"""
Unsupervised anomaly detection using an isolation forest
"""

from typing import Final

import pandas as pd
from sklearn.ensemble import IsolationForest

TRAIN_DATA_PATH: Final[str] = "feature_eng/output/train_data.csv"
PRED_OUTPUT_PATH: Final[str] = "models/predictions/isolation_forest.csv"

data = pd.read_csv(TRAIN_DATA_PATH)
train_X = data[
    [
        "amt",
        "time",
        "day_of_week",
        "src_n_transactions",
        "src_min_amt",
        "src_mean_amt",
        "src_median_amt",
        "src_max_amt",
        "src_ratio_to_min_amt",
        "src_ratio_to_mean_amt",
        "src_ratio_to_median_amt",
        "src_ratio_to_max_amt",
        "dst_n_transactions",
        "dst_min_amt",
        "dst_mean_amt",
        "dst_median_amt",
        "dst_max_amt",
        "dst_ratio_to_min_amt",
        "dst_ratio_to_mean_amt",
        "dst_ratio_to_median_amt",
        "dst_ratio_to_max_amt",
        "src_n_transactions_this_day_of_week",
        "src_prop_transactions_this_day_of_week",
        "src_n_transactions_this_time",
        "src_prop_transactions_this_time",
        "src_n_transactions_this_dst",
        "src_prop_transactions_this_dst",
    ]
]
train_y = data[["is_fraud"]]

isol_forest = IsolationForest(random_state=69, n_jobs=-1)
isol_forest.fit(train_X)
train_pred = pd.DataFrame(
    {
        "tid": data.tid,
        "model": "isolation_forest",
        "anomaly_score": -isol_forest.score_samples(train_X),
    }
)
train_pred["anomaly_score_rank"] = train_pred["anomaly_score"].rank(ascending=False)
train_pred.to_csv(PRED_OUTPUT_PATH, index=False)
print(f"predictions written to '{PRED_OUTPUT_PATH}'")
