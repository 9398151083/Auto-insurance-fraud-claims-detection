import os
import pandas as pd
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline import train_and_export

def run_demo():
    # 1. Load existing output to simulate raw input
    input_csv = r"c:\Users\mahesh\Downloads\Insurance fraud claims detection\investigation_queue.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 2. Drop score columns to simulate raw data
    drop_cols = ["risk_score", "confidence", "priority_rank"]
    df_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    raw_csv_path = r"c:\Users\mahesh\Downloads\Insurance fraud claims detection\raw_claims.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Created simulated raw input: {raw_csv_path}")

    # 3. Run pipeline
    print("Running pipeline...")
    train_and_export(csv_path=raw_csv_path, model_prefix="models/demo_model", top_k=50)

    # 4. Verify output
    output_csv = "investigation_queue.csv"
    if os.path.exists(output_csv):
        print(f"\nSuccess! Output generated: {output_csv}")
        df_out = pd.read_csv(output_csv)
        print(df_out[["policy_number", "risk_score", "confidence", "priority_rank"]].head())
    else:
        print("\nError: Output CSV not found.")

if __name__ == "__main__":
    run_demo()
