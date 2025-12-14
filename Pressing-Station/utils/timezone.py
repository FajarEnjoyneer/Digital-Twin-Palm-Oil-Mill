from datetime import datetime
import pytz

def wib_to_utc_iso(wib_time_str: str) -> str:
    wib = pytz.timezone("Asia/Jakarta")
    utc = pytz.utc

    try:
        dt_wib = datetime.strptime(wib_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("[ERROR] Invalid time format. Use YYYY-MM-DD HH:MM:SS")
        return None

    dt_wib = wib.localize(dt_wib)
    dt_utc = dt_wib.astimezone(utc)

    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
