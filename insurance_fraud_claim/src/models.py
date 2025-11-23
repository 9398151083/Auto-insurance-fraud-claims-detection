"""

Trains two models:
 - IsolationForest (unsupervised anomaly detection)
 - RandomForestClassifier (supervised fraud prediction)
Also includes model saving/loading.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


class FraudPipelineModels:

    def __init__(self, iso_params=None, clf_params=None):

        # Unsupervised anomaly detection
        self.anomaly = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42
        ) if iso_params is None else IsolationForest(**iso_params)

        # Supervised classifier
        self.clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ) if clf_params is None else RandomForestClassifier(**clf_params)

        self.feature_cols = None

   
    # TRAINING
   
    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        self.feature_cols = list(X.columns)

        # Train anomaly model on all data
        self.anomaly.fit(X)

        # Train classifier if labels exist
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            self.clf.fit(X_train, y_train)
            preds = self.clf.predict(X_val)

            print("\n===== RandomForest Classifier Report =====")
            print(classification_report(y_val, preds, digits=4))

            try:
                probas = self.clf.predict_proba(X_val)[:, 1]
                print("ROC AUC:", roc_auc_score(y_val, probas))
            except:
                pass
        else:
            print("No labels provided. Classifier NOT trained.")

    # PREDICTION
   
    def predict(self, X: pd.DataFrame):
        """
        Returns:
         - anomaly_score (0–1)
         - classifier_probability (0–1)
        """

        # IsolationForest decision -> smaller = more anomalous
        raw = self.anomaly.decision_function(X)

        # Normalize to 0–1 (1 = more suspicious)
        a_score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        try:
            c_proba = self.clf.predict_proba(X)[:, 1]
        except:
            c_proba = np.zeros(len(X))

        return {
            "anomaly_score": a_score,
            "clf_proba": c_proba
        }

   
    # SAVE / LOAD MODELS
  
    def save(self, prefix: str):
        joblib.dump(self.anomaly, f"{prefix}_iso.joblib")
        joblib.dump(self.clf, f"{prefix}_clf.joblib")
        joblib.dump(self.feature_cols, f"{prefix}_features.joblib")
        print(f"Models saved with prefix: {prefix}")

    def load(self, prefix: str):
        self.anomaly = joblib.load(f"{prefix}_iso.joblib")
        self.clf = joblib.load(f"{prefix}_clf.joblib")
        self.feature_cols = joblib.load(f"{prefix}_features.joblib")
        print(f"Models loaded from prefix: {prefix}")
