from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG

def build_flux_query(
    bucket: str,
    measurement: str,
    fields: list,
    start: str,
    stop: str,
    window: str = "1s"
) -> str:
    field_filter = " or ".join([f'r["_field"] == "{f}"' for f in fields])

    return f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => {field_filter})
  |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "mean")
'''
