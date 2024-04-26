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

model_names: tuple[str, ...] = (
    "dist_to_dst_clust_median",
    "dist_to_src_clust_median",
    "isolation_forest",
    "local_outlier_factor",
)
model_preds: dict[str, pd.DataFrame] = {}
for model_name in model_names:
    model_preds[model_name] = pd.read_csv(f"models/predictions/{model_name}.csv")
    model_preds[model_name]["anomaly_score_rank"] = model_preds[model_name][
        "anomaly_score"
    ].rank()
    model_preds[model_name] = model_preds[model_name].rename(
        columns={
            "anomaly_score": model_name,
            "anomaly_score_rank": f"{model_name}_rank",
        }
    )
    pred_df = pred_df.merge(
        model_preds[model_name][["tid", model_name, f"{model_name}_rank"]],
        how="left",
        on="tid",
        validate="1:1",
    )

model_names_in_ensemble: list[str] = [
    model_name
    for model_name in pred_df
    if model_name[-5:] == "_rank" and "dist_to_dst_clust_median" not in model_name
]
pred_df["ensemble_agg_avg"] = pred_df[model_names_in_ensemble].mean(axis=1)
print("Ensemble contains the following models:")
for model_name in model_names_in_ensemble:
    print("\t-", model_name.replace("_rank",""))

for model_name in list(model_names) + ["ensemble_agg_avg"]:
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
