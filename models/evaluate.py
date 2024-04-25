"""
Script for evaluating and comparing model performance

Usage:
    $ poetry run python -m models.evaluate
"""

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

RESULTS_OUTPUT_PATH: Final[str] = "models/evaluation_output"

pred_df = pd.read_csv(
    "data/input/simdata.csv",
    usecols=["tid", "is_fraud"],
)
pred_df["is_fraud"] = pred_df["is_fraud"].astype(int)

isolation_forest_df = pd.read_csv("models/predictions/isolation_forest.csv")
isolation_forest_df["anomaly_score_rank"] = isolation_forest_df["anomaly_score"].rank()
isolation_forest_df = isolation_forest_df.rename(
    columns={
        "anomaly_score": "isolation_forest",
        "anomaly_score_rank": "isolation_forest_rank",
    }
)
dist_to_src_clust_median_df = pd.read_csv(
    "models/predictions/dist_to_src_clust_median.csv"
)
dist_to_src_clust_median_df["anomaly_score_rank"] = dist_to_src_clust_median_df[
    "anomaly_score"
].rank()
dist_to_src_clust_median_df = dist_to_src_clust_median_df.rename(
    columns={
        "anomaly_score": "dist_to_src_clust_median",
        "anomaly_score_rank": "dist_to_src_clust_median_rank",
    }
)
pred_df = pred_df.merge(
    isolation_forest_df[["tid", "isolation_forest", "isolation_forest_rank"]],
    how="left",
    on="tid",
)
pred_df = pred_df.merge(
    dist_to_src_clust_median_df[
        ["tid", "dist_to_src_clust_median", "dist_to_src_clust_median_rank"]
    ],
    how="left",
    on="tid",
)
pred_df["ensemble_all_models_agg_avg"] = (
    0.5 * pred_df["isolation_forest_rank"]
    + 0.5 * pred_df["dist_to_src_clust_median_rank"]
)
for model_name in (
    "isolation_forest",
    "dist_to_src_clust_median",
    "ensemble_all_models_agg_avg",
):
    fpr, tpr, thresholds = roc_curve(pred_df["is_fraud"], pred_df[[model_name]])
    auc = roc_auc_score(pred_df["is_fraud"], pred_df[[model_name]])
    plt.plot(fpr, tpr, label="%s ROC (area = %0.2f)" % (model_name, auc))
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.suptitle("Unsupervised Anomaly Detection")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()
