"""

Creates ML-ready features from the cleaned dataset.
Includes:
 - numerical features
 - ratios
 - one-hot encoding for selected categorical variables
"""

import pandas as pd
import numpy as np

def make_features(df: pd.DataFrame, drop_originals: bool = False):
    """
    Creates and returns:
    - processed dataframe with features
    - feature matrix X
    - feature column names list
    """

    dff = df.copy()

    #  NUMERIC FEATURE ENGINEERING 

    # Log-transform total claim amount
    if "total_claim_amount" in dff.columns:
        dff["log_total_claim_amount"] = np.log1p(dff["total_claim_amount"])

    # Claim ratios (injury, property, vehicle)
    for col in ["injury_claim", "property_claim", "vehicle_claim"]:
        if col in dff.columns and "total_claim_amount" in dff.columns:
            dff[f"{col}_ratio"] = dff[col] / (dff["total_claim_amount"].replace({0: 1}))
        elif col in dff.columns:
            dff[f"{col}_ratio"] = dff[col] / (dff[col].max() if dff[col].max() else 1)

    # CATEGORICAL ENCODING 

    
    cat_cols_to_encode = [
        "incident_type",
        "collision_type",
        "incident_severity",
        "policy_state",
        "incident_state",
        "auto_make",
        "insured_education_level",
        "insured_occupation",
        "insured_relationship"
    ]

    all_cat_features = []

    for col in cat_cols_to_encode:
        if col in dff.columns:
            # Keep top 10 values; others become "OTHER"
            top_vals = dff[col].value_counts().nlargest(10).index
            dff[col] = dff[col].where(dff[col].isin(top_vals), "OTHER")

            # One-hot encode
            dummies = pd.get_dummies(dff[col], prefix=col)
            dff = pd.concat([dff, dummies], axis=1)
            all_cat_features.extend(dummies.columns)

    # NUMERIC FEATURES TO KEEP 

    numeric_features = [
        "log_total_claim_amount",
        "months_as_customer",
        "age",
        "policy_annual_premium",
        "number_of_vehicles_involved",
        "bodily_injuries",
        "witnesses",
        "incident_hour_of_the_day",
        "injury_claim_ratio",
        "property_claim_ratio",
        "vehicle_claim_ratio",
        "claim_month",
        "claim_dayofweek"
    ]

    final_numeric_features = [c for c in numeric_features if c in dff.columns]

    # Combine numeric + categorical
    feature_cols = final_numeric_features + all_cat_features

    # Create final feature matrix
    X = dff[feature_cols].fillna(0)

    return dff, X, feature_cols
