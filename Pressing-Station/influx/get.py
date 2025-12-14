import pandas as pd
from influxdb_client import InfluxDBClient
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG


def fetch_data(flux_query: str) -> pd.DataFrame:
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        df = client.query_api().query_data_frame(flux_query)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        return df
