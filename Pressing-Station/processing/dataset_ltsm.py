import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIG
# ==========================================
WINDOW_SIZE = 60   # 60 data sebelumnya
STEP_SIZE = 1
TARGET = "SP6_press"

# ==========================================
# LOAD FEATURE ENGINEERED DATA
# ==========================================
df = pd.read_csv("dataset_feature_engineered.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

# ==========================================
# PILIH Fitur untuk LSTM
# (gunakan fitur yang sudah dibuat sebelumnya)
# ==========================================
feature_cols = [
    "SP6_current",
    "SP6_press",
    "current_roll_mean_5",
    "press_roll_mean_5",
    "current_delta",
    "press_delta",
]

df = df[feature_cols].dropna()

# ==========================================
# NORMALISASI
# ==========================================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

# ==========================================
# BUILD WINDOWED DATASET UNTUK LSTM
# X shape => (samples, window, features)
# y shape => (samples,)
# ==========================================
X, y = [], []

for i in range(0, len(scaled_df) - WINDOW_SIZE, STEP_SIZE):
    window = scaled_df.iloc[i:i + WINDOW_SIZE].values
    target_value = scaled_df.iloc[i + WINDOW_SIZE][TARGET]
    
    X.append(window)
    y.append(target_value)

X = np.array(X)
y = np.array(y)

print("LSTM dataset shape:", X.shape, y.shape)

# ==========================================
# SAVE DATASET
# ==========================================
np.savez("dataset_lstm.npz", X=X, y=y, feature_names=np.array(feature_cols))

print("dataset_lstm.npz berhasil dibuat!")
