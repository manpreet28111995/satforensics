import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://network.satnogs.org/api/observations/"
KEEP = [
    "id", "start", "end", "ground_station", "norad_cat_id", "status", "status_certainty",
    "waterfall", "waterfall_status", "station_lat", "station_lng", "station_alt",
    "rise_azimuth", "set_azimuth", "max_altitude", "transmitter_uuid", "transmitter_type",
    "transmitter_mode", "transmitter_baud", "transmitter_invert", "transmitter_downlink_low",
    "transmitter_downlink_high", "transmitter_downlink_drift", "transmitter_uplink_low",
    "observation_frequency", "tle0", "tle1", "tle2", "client_version", "observer", "sat_id",
]


def fetch_json(url, retries=6, timeout=60):
    delay = 5.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "satforensics-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.headers.get("Link", "")
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return None, ""


def next_url(link_header):
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part[part.index("<") + 1:part.index(">")]
    return None


def query_window(start, end, pages=2):
    params = urllib.parse.urlencode({"format": "json", "start": start, "end": end})
    url = BASE + "?" + params
    out = []
    for _ in range(pages):
        data, link = fetch_json(url)
        if not data:
            break
        out.extend(data)
        url = next_url(link)
        if not url:
            break
    return out


def slim(obs):
    return {k: obs.get(k) for k in KEEP}
