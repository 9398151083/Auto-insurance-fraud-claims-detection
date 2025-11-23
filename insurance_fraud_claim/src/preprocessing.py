"""
preprocessing.py
Handles missing values, fixes categorical/text fields, and creates simple derived columns.
"""

import pandas as pd
import numpy as np

def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the raw DataFrame loaded from CSV.
    """

    dff = df.copy()

    # Drop rows missing incident_date
    if "incident_date" in dff.columns:
        dff = dff[~dff["incident_date"].isna()].copy()

    # Fill missing categorical fields
    fill_cats = [
        "policy_state", "incident_state", "incident_city",
        "policy_number", "insured_zip", "collision_type",
        "incident_type", "incident_severity", "auto_make", "auto_model"
    ]
    for col in fill_cats:
        if col in dff.columns:
            dff[col] = dff[col].fillna("UNKNOWN").astype(str).str.strip()

    # Fill missing numeric fields with 0
    num_cols = dff.select_dtypes(include=[np.number]).columns
    dff[num_cols] = dff[num_cols].fillna(0)

    # Clean string columns
    str_cols = dff.select_dtypes(include=["object"]).columns
    for c in str_cols:
        dff[c] = dff[c].astype(str).str.strip()

    # Derived (feature-friendly) columns
    if "incident_date" in dff.columns:
        dff["claim_month"] = dff["incident_date"].dt.month
        dff["claim_dayofweek"] = dff["incident_date"].dt.dayofweek

    # Ensure integer values where needed
    int_cols = ["incident_hour_of_the_day", "number_of_vehicles_involved",
                "bodily_injuries", "witnesses"]
    for col in int_cols:
        if col in dff.columns:
            dff[col] = pd.to_numeric(dff[col], errors="coerce").fillna(0).astype(int)

    return dff
