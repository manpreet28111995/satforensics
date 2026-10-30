import datetime as dt
import math

import numpy as np
from sgp4.api import Satrec, jday

C_KM_S = 299792.458
WGS_A = 6378.137
WGS_F = 1.0 / 298.257223563
WGS_E2 = WGS_F * (2.0 - WGS_F)
EARTH_ROT = 7.2921158553e-5


def parse_time(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def gmst_rad(jd, fr):
    d = (jd - 2451545.0) + fr
    deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return math.radians(deg)


def station_ecef(lat_deg, lng_deg, alt_m):
    lat, lng = math.radians(lat_deg), math.radians(lng_deg)
    n = WGS_A / math.sqrt(1.0 - WGS_E2 * math.sin(lat) ** 2)
    h = alt_m / 1000.0
    return np.array([
        (n + h) * math.cos(lat) * math.cos(lng),
        (n + h) * math.cos(lat) * math.sin(lng),
        (n * (1.0 - WGS_E2) + h) * math.sin(lat),
    ])


def enu_basis(lat_deg, lng_deg):
    lat, lng = math.radians(lat_deg), math.radians(lng_deg)
    east = np.array([-math.sin(lng), math.cos(lng), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lng), -math.sin(lat) * math.sin(lng), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lng), math.cos(lat) * math.sin(lng), math.sin(lat)])
    return east, north, up


def track(rec, steps=64):
    sat = Satrec.twoline2rv(rec["tle1"], rec["tle2"])
    t0, t1 = parse_time(rec["start"]), parse_time(rec["end"])
    span = (t1 - t0).total_seconds()
    obs = station_ecef(rec["station_lat"], rec["station_lng"], rec["station_alt"] or 0.0)
    east, north, up = enu_basis(rec["station_lat"], rec["station_lng"])
    out = np.zeros((steps, 6), dtype=np.float64)
    for k in range(steps):
        t = t0 + dt.timedelta(seconds=span * k / (steps - 1))
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        err, pos, vel = sat.sgp4(jd, fr)
        if err != 0:
            return None
        th = gmst_rad(jd, fr)
        c, s = math.cos(th), math.sin(th)
        rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        p = rot @ np.array(pos)
        v = rot @ np.array(vel) - np.cross(np.array([0.0, 0.0, EARTH_ROT]), p)
        rho = p - obs
        rng = np.linalg.norm(rho)
        rr = float(np.dot(rho, v) / rng)
        e_, n_, u_ = float(rho @ east), float(rho @ north), float(rho @ up)
        elev = math.degrees(math.asin(u_ / rng))
        az = math.atan2(e_, n_)
        out[k] = (elev, math.sin(az), math.cos(az), rng, rr, span * k / (steps - 1))
    return out


def frequency_hz(rec):
    f = rec.get("observation_frequency") or rec.get("transmitter_downlink_low") or 0.0
    return float(f)
