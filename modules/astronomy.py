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


import re
import numpy as np

def ra_dec_to_radians(radec, is_ra=True):
    # Convertimos a string por si acaso entra un float/int directo
    radec_str = str(radec).strip()
    
    # Busca todos los números (incluyendo decimales y signos)
    parts = re.findall(r"[-+]?\d*\.\d+|\d+", radec_str)
    parts = list(map(float, parts))
    
    if len(parts) == 0:
        raise ValueError(f"No se encontraron números en: {radec}")

    # CASO 1: Ya viene en grados decimales (solo un número)
    if len(parts) == 1:
        decimal_value = parts[0]
        # Si es RA y viene en un solo número, asumimos que son GRADOS.
        # (Si fueran horas, habría que multiplicarlo por 15 aquí).
        degrees = decimal_value
        
    # CASO 2: Formato sexagesimal (H/D, M, S)
    else:
        val, m, s = parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0
        
        # Calculamos el valor absoluto decimal
        decimal_value = abs(val) + m / 60.0 + s / 3600.0
        
        if is_ra:
            # Ascensión Recta: 1h = 15 grados
            degrees = decimal_value * 15
        else:
            # Declinación: Respetamos el signo del primer componente
            degrees = decimal_value if val >= 0 else -decimal_value
        
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

    times_sec = np.array([(t - times_utc[0]).total_seconds() for t in times_utc])

    return times_utc, H, lst_rad, times_sec
