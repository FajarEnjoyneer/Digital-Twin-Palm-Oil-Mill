from influxdb_client import InfluxDBClient
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG

def list_buckets() -> list:
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        buckets = client.buckets_api().find_buckets().buckets
        return [b.name for b in buckets]


def list_measurements(bucket: str) -> list:
    query = f'''
import "influxdata/influxdb/schema"
schema.measurements(bucket: "{bucket}")
'''
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        tables = client.query_api().query(query)
        measurements = []
        for table in tables:
            for record in table.records:
                measurements.append(record.get_value())
        return measurements


def list_fields(bucket: str, measurement: str) -> list:
    query = f'''
import "influxdata/influxdb/schema"
schema.fieldKeys(
  bucket: "{bucket}",
  predicate: (r) => r._measurement == "{measurement}"
)
'''
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        tables = client.query_api().query(query)
        fields = []
        for table in tables:
            for record in table.records:
                fields.append(record.get_value())
        return fields
