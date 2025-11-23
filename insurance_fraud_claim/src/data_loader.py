import os
import pandas as pd

def load_claims_csv(path: str) -> pd.DataFrame:
    """
    Load the insurance claims CSV.
    Automatically parses 'incident_date' and converts numeric columns.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    # Load CSV
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Parse the date column
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    else:
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if not date_cols:
            raise ValueError("No 'incident_date' column found in CSV.")
        df["incident_date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")

    # Convert numeric columns safely
    numeric_cols = [
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
        "months_as_customer", "age", "policy_annual_premium",
        "number_of_vehicles_involved", "bodily_injuries", "witnesses",
        "incident_hour_of_the_day"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fix the label column
    if "fraud_reported" in df.columns:
        df["fraud_reported"] = (
            df["fraud_reported"]
            .astype(str)
            .str.upper()
            .map({"Y": 1, "N": 0})
        ).fillna(0).astype(int)

    # Remove unnamed or irrelevant columns
    drop_cols = [c for c in df.columns if c.startswith("Unnamed") or c.startswith("_c")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df.reset_index(drop=True)
