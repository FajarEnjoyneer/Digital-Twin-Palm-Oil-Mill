import os

INFLUX_HOST = os.getenv("INFLUX_HOST", "localhost")
INFLUX_PORT = int(os.getenv("INFLUX_PORT", 8086))
INFLUX_URL = os.getenv("INFLUX_URL", f"http://{INFLUX_HOST}:{INFLUX_PORT}")

INFLUX_ORG = os.getenv("INFLUX_ORG", "725076b7415d5012")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "GUJQRZxBWmHd9v0AP42d4wfbfLFjB1_nm13STvkk-18o-d9BOQgTmPIFaspx3CHRdtTlYGrCVgYFcy1RMcc99A==")

DEFAULT_WINDOW = os.getenv("INFLUX_DEFAULT_WINDOW", "1s")
DEFAULT_OUTPUT_DIR_FEATURE = os.getenv("PIPELINE_OUTPUT_DIR", "output/dataset_feature")
DEFAULT_OUTPUT_DIR_LSTM = os.getenv("PIPELINE_OUTPUT_DIR", "output/dataset_lstm")