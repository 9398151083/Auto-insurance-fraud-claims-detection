"""
evaluation.py
Provides helper functions to evaluate classifier performance.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ------------------------------------------------------------
# CLASSIFIER EVALUATION
# ------------------------------------------------------------
def evaluate_classifier(y_true, y_pred, y_proba=None):
    """
    Prints classification metrics:
      - precision, recall, f1
      - confusion matrix
      - ROC-AUC (if probabilities exist)
    """

    print("\n===== CLASSIFIER EVALUATION =====")
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            print("\nROC-AUC:", round(auc, 4))
        except Exception:
            print("ROC-AUC could not be calculated.")


# ------------------------------------------------------------
# TOP-K SCORE THRESHOLDING
# ------------------------------------------------------------
def threshold_topk(df_with_scores, k=100):
    """
    Takes a dataframe containing a `risk_score` column and returns top-k rows.
    """

    if "risk_score" not in df_with_scores.columns:
        raise ValueError("DataFrame must contain 'risk_score' column")

    return df_with_scores.sort_values("risk_score", ascending=False).head(k)
