"""
utils.py
Utility functions such as saving CSV outputs.
"""

import pandas as pd

def save_csv(df: pd.DataFrame, path: str):
    """
    Saves DataFrame to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Saved CSV to: {path}")
