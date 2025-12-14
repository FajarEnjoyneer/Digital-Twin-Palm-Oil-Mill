import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_lstm_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print(f"[INFO] Dataset loaded: X={X.shape}, y={y.shape}")
    return X, y

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
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

    return model

def train_lstm(
    dataset_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    epochs: int = 100,
    batch_size: int = 32,
    plot: bool = True
):
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_lstm_dataset(dataset_path)

    train_size = int(len(X) * train_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    model.summary()

    best_model_path = os.path.join(output_dir, "lstm_best_model.keras")
    final_model_path = os.path.join(output_dir, "lstm_final_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(final_model_path)
    print(f"[DONE] Model saved: {final_model_path}")

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test MSE: {test_loss:.4f}")
    print(f"[RESULT] Test MAE: {test_mae:.4f}")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("LSTM Training Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "test_mse": test_loss,
        "test_mae": test_mae,
        "best_model": best_model_path,
        "final_model": final_model_path
    }
