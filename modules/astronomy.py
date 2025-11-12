"""
Funciones astronómicas y de tiempo sidéreo
"""


import numpy as np
from datetime import datetime, timedelta, timezone
import juliandate as jd
from .utils import degree_to_time

def local_sidereal_time(longitude=-70.76, utc=None, single=True):
    if utc is None:
        now = datetime.now(timezone.utc)
    else:
        now = utc

    jd_now = jd.from_gregorian(now.year, now.month, now.day, now.hour, now.minute, now.second)
    T = (jd_now - 2451545.0) / 36525
    theta = 280.46061837 + 360.98564736629 * (jd_now - 2451545) + (0.000387933 * T * T) - (T * T * T / 38710000.0)
    deg = theta % 360 + longitude

    h, m, s = degree_to_time(deg)
    rad = np.deg2rad(deg)
  
    return rad if single else (deg, rad, h, m, s)


def ra_dec_to_radians(radec, is_ra=True):
    h, m, s = map(float, radec.split(':'))
    value = abs(h) + m / 60 + s / 3600
    degrees = value * 15 if is_ra else (value if h >= 0 else -value)
    return np.deg2rad(degrees)


def H_range(ra_rad, utc_start, utc_end, longitude=-70.76, step_minutes=5):
    if isinstance(utc_start, str):
        utc_start = datetime.fromisoformat(utc_start).replace(tzinfo=timezone.utc)
    if isinstance(utc_end, str):
        utc_end = datetime.fromisoformat(utc_end).replace(tzinfo=timezone.utc)

    n_steps = int((utc_end - utc_start).total_seconds() / 60 / step_minutes) + 1
    times_utc = [utc_start + timedelta(minutes=i * step_minutes) for i in range(n_steps)]

    lst_rad = np.array([
        local_sidereal_time(longitude=longitude, utc=t, single=True)
        for t in times_utc
    ])

    H = (lst_rad - ra_rad + np.pi) % (2 * np.pi) - np.pi
    return times_utc, H, lst_rad
