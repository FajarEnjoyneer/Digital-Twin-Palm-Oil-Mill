import os
import matplotlib.pyplot as plt
import pandas as pd

def clean_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"_time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna().sort_values("timestamp")
    return df

def run_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.plot(x="timestamp", figsize=(12,4))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timeseries.png")
    plt.close()

    print("\nCorrelation:")
    print(df.corr(numeric_only=True))
