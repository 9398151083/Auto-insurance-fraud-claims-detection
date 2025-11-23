"""
pipeline.py
Runs the complete fraud detection pipeline:

1. Load CSV
2. Preprocess data
3. Feature engineering
4. Network graph scoring
5. Train IsolationForest + RandomForest
6. Generate fraud risk scores
7. Produce investigation queue
8. Save models + output CSV

Run from terminal:
    python src/pipeline.py --csv "path/to/insurance_claims.csv"
"""

import argparse
import os

from data_loader import load_claims_csv
from preprocessing import preprocess_basic
from feature_engineering import make_features
from network_analysis import build_light_graph, compute_network_score
from models import FraudPipelineModels
from scoring import compute_risk, build_queue
from utils import save_csv



# MAIN FUNCTION

def train_and_export(csv_path: str, model_prefix: str = "models/fraud_model", top_k: int = 200):

    print("\n===== STEP 1: LOADING DATA =====")
    df = load_claims_csv(csv_path)

    print("\n===== STEP 2: PREPROCESSING =====")
    df_prep = preprocess_basic(df)

    print("\n===== STEP 3: FEATURE ENGINEERING =====")
    df_feat, X, feature_cols = make_features(df_prep)

    print("\n===== STEP 4: NETWORK ANALYSIS =====")
    G = build_light_graph(df_feat)
    network_scores = compute_network_score(df_feat, G)

    print("\n===== STEP 5: MODEL TRAINING =====")
    y = None
    if "fraud_reported" in df_feat.columns:
        y = df_feat["fraud_reported"]

    model = FraudPipelineModels()
    model.fit(X, y)

    print("\n===== STEP 6: MODEL PREDICTION =====")
    preds = model.predict(X)

    print("\n===== STEP 7: RISK SCORING =====")
    risk_scores, confidence = compute_risk(
        preds["anomaly_score"],
        preds["clf_proba"],
        network_scores
    )

    queue = build_queue(df_feat, risk_scores, confidence, top_k=top_k)

    print("\n===== STEP 8: SAVING RESULTS =====")

    # Ensure model folder exists
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    model.save(model_prefix)
    save_csv(queue, "investigation_queue.csv")

    print("\n=== PIPELINE COMPLETE ===")
    return queue



# COMMAND-LINE ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to insurance_claims CSV file")
    parser.add_argument("--out_prefix", default="models/fraud_model", help="Model save prefix")
    parser.add_argument("--topk", type=int, default=200, help="Top suspicious cases to output")

    args, _ = parser.parse_known_args()

    train_and_export(
        csv_path=args.csv,
        model_prefix=args.out_prefix,
        top_k=args.topk
    )

