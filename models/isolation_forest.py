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
        "src_ratio_to_min_amt",
        "src_ratio_to_mean_amt",
        "src_ratio_to_median_amt",
        "src_ratio_to_max_amt",
        "prop_transactions_this_day_of_week",
        "prop_transactions_this_time",
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
train_pred.to_csv(PRED_OUTPUT_PATH, index=False)
