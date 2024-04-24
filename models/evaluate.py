"""
Script for evaluating and comparing model performance

Usage:
    $ poetry run python -m models.evaluate
"""

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, RocCurveDisplay

RESULTS_OUTPUT_PATH: Final[str] = "models/evaluation_output"

y_true_df = pd.read_csv(
    "data/input/simdata.csv",
    usecols=["tid", "is_fraud"],
)
y_true_df["is_fraud"] = y_true_df["is_fraud"].astype(int)

isolation_forest_df = pd.read_csv("models/predictions/isolation_forest.csv")

# ROC AUC curve #
# fpr, tpr, thresholds = roc_curve(
#    y_true=y_true_df["is_fraud"], y_score=isolation_forest_df["anomaly_score"]
# )
roc_auc_curve_display = RocCurveDisplay.from_predictions(
    y_true=y_true_df["is_fraud"],
    y_pred=isolation_forest_df["anomaly_score"],
    plot_chance_level=True,
)
roc_auc_curve_display.plot()
plt.show()
