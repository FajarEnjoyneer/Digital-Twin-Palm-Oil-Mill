import os
import sys
from influx.get import fetch_data
from influx.query_builder import build_flux_query
from influx.metadata import list_buckets, list_measurements, list_fields
from processing.eda import clean_and_pivot, run_eda
from processing.feature import make_features, save_features
from utils.timezone import wib_to_utc_iso
from config import DEFAULT_WINDOW, DEFAULT_OUTPUT_DIR


def choose_from_list(title: str, options: list) -> str:
    if not options:
        print(f"[ERROR] No {title} found")
        sys.exit(1)

    print(f"\nSelect {title}:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")

    while True:
        try:
            idx = int(input("Number: "))
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print("Invalid number")
        except ValueError:
            print("Enter a number")


def main():
    print("\n Get Data ")

    bucket = choose_from_list("bucket", list_buckets())
    measurement = choose_from_list("measurement", list_measurements(bucket))

    fields = list_fields(bucket, measurement)
    print("\nAvailable fields:")
    print(", ".join(fields))

    chosen_fields = input("Fields (comma separated): ").split(",")
    chosen_fields = [f.strip() for f in chosen_fields if f.strip()]

    invalid = set(chosen_fields) - set(fields)
    if invalid:
        print(f"[ERROR] Invalid fields: {invalid}")
        sys.exit(1)

    print("\nTime format: YYYY-MM-DD HH:MM:SS (WIB)")
    start_wib = input("Start time: ")
    stop_wib = input("Stop time : ")

    start_utc = wib_to_utc_iso(start_wib)
    stop_utc = wib_to_utc_iso(stop_wib)

    window = input(f"Window [{DEFAULT_WINDOW}]: ") or DEFAULT_WINDOW

    query = build_flux_query(
        bucket=bucket,
        measurement=measurement,
        fields=chosen_fields,
        start=start_utc,
        stop=stop_utc,
        window=window
    )

    print("Fetching data...")
    raw_df = fetch_data(query)

    if raw_df.empty:
        print("No data found")
        sys.exit(0)

    print("Cleaning data...")
    df_clean = clean_and_pivot(raw_df)

    print("Running EDA...")
    run_eda(df_clean, output_dir=DEFAULT_OUTPUT_DIR)

    print("Generating features...")
    df_fe = make_features(df_clean)

    output_path = os.path.join(DEFAULT_OUTPUT_DIR, "dataset_feature_engineered.csv")
    save_features(df_fe, output_path)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
