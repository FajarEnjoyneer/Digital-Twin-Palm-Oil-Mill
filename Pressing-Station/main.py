import os
import sys
import glob
import subprocess
import pandas as pd
from datetime import datetime
from typing import List

from influx.get import fetch_data
from influx.query_builder import build_flux_query
from influx.metadata import list_buckets, list_measurements, list_fields
from processing.eda import clean_and_pivot, run_eda
from processing.feature import make_features, save_features
from processing.dataset_ltsm import create_lstm_dataset
from utils.timezone import wib_to_utc_iso
from config import DEFAULT_WINDOW, DEFAULT_OUTPUT_DIR_FEATURE, DEFAULT_OUTPUT_DIR_LSTM
from training.lstm import train_lstm

# Utils
def choose_from_list(title: str, options: List[str]) -> str:
    if not options:
        print(f"[ERROR] No {title} found")
        sys.exit(1)

    print(f"\nSelect {title}:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")

    while True:
        try:
            idx = int(input("Number: ").strip())
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print("Invalid number")
        except ValueError:
            print("Enter a number")


def menu() -> str:
    print("\n===== Digital Twin – Pressing Station =====")
    print("1. Create dataset_feature")
    print("2. Create dataset_lstm")
    print("3. Training LSTM")
    print("0. Exit")
    return input("Choose menu: ").strip()


def ask_time(prompt: str) -> str:
    """
    Ask user for WIB time string until it is parseable.
    Expected format: YYYY-MM-DD HH:MM:SS
    """
    while True:
        s = input(prompt).strip()
        try:
            datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return s
        except ValueError:
            print("Invalid format. Use: YYYY-MM-DD HH:MM:SS (WIB). Try again.")


def choose_file_from_dir(dirpath: str, pattern: str = "*.csv") -> str:
    files = sorted(glob.glob(os.path.join(dirpath, pattern)), key=os.path.getmtime, reverse=True)
    if not files:
        return ""
    print("\nAvailable feature CSV files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {os.path.basename(f)}  (modified: {datetime.fromtimestamp(os.path.getmtime(f))})")
    print("0. Use latest file")
    while True:
        sel = input("Choose file number (or press Enter for latest): ").strip()
        if sel == "" or sel == "0":
            return files[0]
        try:
            idx = int(sel)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            pass
        print("Invalid selection")


# Step 1: Create dataset_feature
def create_dataset_feature():
    print("\n[STEP 1] Create dataset_feature")

    try:
        bucket = choose_from_list("bucket", list_buckets())
        measurement = choose_from_list("measurement", list_measurements(bucket))
    except Exception as e:
        print(f"[ERROR] Failed to list buckets/measurements: {e}")
        return

    try:
        fields = list_fields(bucket, measurement)
    except Exception as e:
        print(f"[ERROR] Failed to list fields: {e}")
        return

    print("\nAvailable fields:")
    print(", ".join(fields))

    chosen_fields = input("Fields (comma separated): ").split(",")
    chosen_fields = [f.strip() for f in chosen_fields if f.strip()]
    if not chosen_fields:
        print("[ERROR] No fields selected")
        return

    invalid = set(chosen_fields) - set(fields)
    if invalid:
        print(f"[ERROR] Invalid fields: {invalid}")
        return

    print("\nTime format: YYYY-MM-DD HH:MM:SS (WIB)")
    start_wib = ask_time("Start time: ")
    stop_wib = ask_time("Stop time : ")

    start_utc = wib_to_utc_iso(start_wib)
    stop_utc = wib_to_utc_iso(stop_wib)

    window = input(f"Window [{DEFAULT_WINDOW}]: ").strip() or DEFAULT_WINDOW

    query = build_flux_query(
        bucket=bucket,
        measurement=measurement,
        fields=chosen_fields,
        start=start_utc,
        stop=stop_utc,
        window=window
    )

    print("Fetching data...")
    try:
        raw_df = fetch_data(query)
    except Exception as e:
        print(f"[ERROR] fetch_data failed: {e}")
        return

    if raw_df is None or raw_df.empty:
        print("No data found for the given query/time range.")
        return

    try:
        print("Cleaning data...")
        df_clean = clean_and_pivot(raw_df)
    except Exception as e:
        print(f"[ERROR] clean_and_pivot failed: {e}")
        return

    try:
        print("Running EDA...")
        os.makedirs(DEFAULT_OUTPUT_DIR_FEATURE, exist_ok=True)
        run_eda(df_clean, output_dir=DEFAULT_OUTPUT_DIR_FEATURE)
    except Exception as e:
        print(f"[WARN] run_eda returned error: {e} (continuing)")

    try:
        print("Generating features...")
        df_fe = make_features(df_clean)
    except Exception as e:
        print(f"[ERROR] make_features failed: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DEFAULT_OUTPUT_DIR_FEATURE, exist_ok=True)
    output_csv = os.path.join(DEFAULT_OUTPUT_DIR_FEATURE, f"dataset_feature_{timestamp}.csv")
    try:
        save_features(df_fe, output_csv)
    except Exception as e:
        print(f"[ERROR] save_features failed: {e}")
        return

    print(f"[DONE] Feature dataset saved: {output_csv}")


# Step 2: Create dataset_lstm
def create_dataset_lstm_menu():
    print("\n[STEP 2] Create dataset_lstm")

    feature_dir = DEFAULT_OUTPUT_DIR_FEATURE
    latest_csv = choose_file_from_dir(feature_dir)
    if not latest_csv:
        print(f"[ERROR] No feature CSV files found in {feature_dir}")
        return

    print(f"[INFO] Using feature file: {os.path.basename(latest_csv)}")

    try:
        df = pd.read_csv(latest_csv, parse_dates=["timestamp"])
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        print("[ERROR] No numeric columns found in feature CSV")
        return

    print("\nAvailable numeric columns:")
    for i, c in enumerate(numeric_cols, 1):
        print(f"{i}. {c}")
    print("0. Cancel")

    while True:
        choice = input("Choose target column (enter name or number): ").strip()
        if choice == "0" or choice == "":
            print("Canceled.")
            return
        # if numeric input map to index
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(numeric_cols):
                target = numeric_cols[idx - 1]
                break
            else:
                print("Invalid number")
                continue
        else:
            if choice in numeric_cols:
                target = choice
                break
            else:
                print("Invalid column name. Try again.")

    try:
        os.makedirs(DEFAULT_OUTPUT_DIR_LSTM, exist_ok=True)
        create_lstm_dataset(
            df,
            output_dir=DEFAULT_OUTPUT_DIR_LSTM,
            target_column=target
        )
    except Exception as e:
        print(f"[ERROR] create_lstm_dataset failed: {e}")
        return

    print("[DONE] LSTM dataset created")


# Step 3: Train LSTM
def train_lstm_menu():
    print("\n[STEP 3] Training LSTM")

    npz_files = glob.glob(os.path.join(DEFAULT_OUTPUT_DIR_LSTM, "*.npz"))
    if not npz_files:
        print("[ERROR] No LSTM dataset found")
        return

    latest_npz = max(npz_files, key=os.path.getmtime)
    print(f"[INFO] Using dataset: {os.path.basename(latest_npz)}")

    train_lstm(
        dataset_path=latest_npz,
        output_dir=os.path.join(DEFAULT_OUTPUT_DIR_LSTM, "model")
    )


# Main menu loop
def main():
    while True:
        choice = menu()
        if choice == "1":
            create_dataset_feature()
        elif choice == "2":
            create_dataset_lstm_menu()
        elif choice == "3":
            train_lstm_menu()
        elif choice == "0":
            print("Exit.")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
