"""
scoring.py
Combines model outputs into a final risk score for fraud detection.
Also builds an investigation priority queue.
"""

import numpy as np
import pandas as pd



# CALCULATE RISK SCORE

def compute_risk(anomaly_score, clf_proba, network_score, weights=None):
    """
    Returns:
      - final risk score (0–1)
      - model confidence (agreement between anomaly + classifier)
    """

    if weights is None:
        weights = {
            "anomaly": 0.45,
            "clf": 0.45,
            "network": 0.10
        }

    # Normalize all inputs to 0–1
    def norm(x):
        x = np.array(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    a = norm(anomaly_score)
    c = norm(clf_proba)
    n = norm(network_score)

    # Weighted sum
    score = (
        weights["anomaly"] * a +
        weights["clf"] * c +
        weights["network"] * n
    )

    # Confidence = agreement between classifier and anomaly
    confidence = 1 - np.abs(a - c)

    return score, confidence



# BUILD INVESTIGATION QUEUE

def build_queue(df: pd.DataFrame, score_arr, conf_arr, top_k=200):
    """
    Sorts claims by:
      1. Highest risk score
      2. Highest confidence
      3. Highest claim amount (tie-breaker)
    """

    out = df.copy()

    out["risk_score"] = score_arr
    out["confidence"] = conf_arr

    # Step 1 → base sort by risk and confidence
    sort_columns = ["risk_score", "confidence"]
    ascending_order = [False, False]

    # Step 2 → add claim amount if column exists
    if "total_claim_amount" in out.columns:
        sort_columns.append("total_claim_amount")
        ascending_order.append(False)
    elif "claim_amount" in out.columns:
        sort_columns.append("claim_amount")
        ascending_order.append(False)

    # Step 3 → final sorted output
    out = out.sort_values(sort_columns, ascending=ascending_order).reset_index(drop=True)

    out["priority_rank"] = out.index + 1

    return out.head(top_k)
