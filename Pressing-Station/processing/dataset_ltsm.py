import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Optional, Sequence
WINDOW_SIZE = 60
STEP_SIZE = 1
OUTPUT_FILENAME = "dataset_lstm.npz"
SCALER_FILENAME = "dataset_lstm_scaler.gz"
def create_lstm_dataset(
        df: pd.DataFrame,
    output_dir: str,
    target_column: str,
    feature_columns: Optional[Sequence[str]] = None,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
):
    print("[INFO] Creating LSTM dataset")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataset must have 'timestamp' column or DatetimeIndex")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    if feature_columns is None:

        feature_columns = (
                df.select_dtypes(include=["float", "int"])
              .columns
              .drop(target_column)
              .tolist()
        )
    if not feature_columns:
        raise ValueError("No feature columns available for LSTM")
    print(f"[INFO] Target  : {target_column}")
    print(f"[INFO] Features: {feature_columns}")
    df_model = df[feature_columns + [target_column]].dropna()
    if len(df_model) <= window_size:
        raise ValueError("Not enough data for LSTM windowing")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)
    scaled_df = pd.DataFrame(
            scaled,
        columns=feature_columns + [target_column],
        index=df_model.index
    )
    X, y = [], []
    for i in range(0, len(scaled_df) - window_size, step_size):
        X.append(
                scaled_df.iloc[i:i + window_size][feature_columns].values
        )
        y.append(
                scaled_df.iloc[i + window_size][target_column]
        )
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, OUTPUT_FILENAME)
    scaler_path = os.path.join(output_dir, SCALER_FILENAME)
    np.savez_compressed(
            dataset_path,
        X=X,
        y=y,
        feature_names=np.array(feature_columns, dtype=object),
        target_name=target_column
    )
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Dataset saved : {dataset_path}")
    print(f"[INFO] Scaler saved  : {scaler_path}")
    return dataset_path
