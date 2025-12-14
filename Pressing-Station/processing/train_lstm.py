import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATASET
# ==========================================
data = np.load("dataset_lstm.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

print("Dataset loaded:", X.shape, y.shape)

# ==========================================
# TRAIN/TEST SPLIT (time-series)
# ==========================================
train_ratio = 0.8
train_size = int(len(X) * train_ratio)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# ==========================================
# BUILD LSTM MODEL
# ==========================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("lstm_best_model.keras", monitor="val_loss", save_best_only=True)
]

# ==========================================
# TRAINING
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================
model.save("lstm_final_model.keras")
print("Model saved as lstm_final_model.keras")

# ==========================================
# EVALUATION
# ==========================================
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

# ==========================================
# PLOT TRAINING LOSS
# ==========================================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("LSTM Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
