import pandas as pd
import numpy as np

INPUT = "data1.csv"
OUTPUT = "data1_clean.csv"

def impute_mean_mode(input_csv=INPUT, output_csv=OUTPUT, verbose=True):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    if verbose:
        print(f"Loaded {input_csv} with shape {df.shape}")

    missing_before = df.isna().sum()
    if verbose:
        print("\nMissing counts (before) - top 20:")
        print(missing_before[missing_before > 0].sort_values(ascending=False).head(20))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for c in numeric_cols:
        mean_val = df[c].mean()
        if np.isnan(mean_val):
            df[c] = df[c].fillna(0)
            if verbose:
                print(f"Numeric column '{c}' all-NA -> filled with 0")
        else:
            df[c] = df[c].fillna(mean_val)

    for c in categorical_cols:
        s = df[c].astype(object)
        if s.dropna().empty:
            df[c] = s.fillna("missing").astype(str)
            if verbose:
                print(f"Categorical column '{c}' all-NA -> filled with 'missing'")
            continue
        try:
            mode_val = s.mode(dropna=True)
            if not mode_val.empty:
                fill_val = mode_val.iloc[0]
            else:
                fill_val = "missing"
        except Exception:
            fill_val = "missing"
        df[c] = s.fillna(fill_val).astype(str)

    missing_after = df.isna().sum()

    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\nSaved to: {output_csv}")

    return df

if __name__ == "__main__":
    impute_mean_mode()
