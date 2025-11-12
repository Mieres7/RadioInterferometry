"""
Defines a full simulation of diferent kinds.
"""

import numpy as np

from modules.coords import baselines, enu_to_altaz, hor_to_eq, eq_to_uvw
from modules.catalogs import SOURCES_CATALOG, select_vla_frequencies
from modules.astronomy import ra_dec_to_radians, H_range
from modules.interferometry import generate_random_sources, uvw_to_lambda_range, visibilities_from_sources
from modules.utils import read_cfg_to_enu

def visibilities_simulation(config):
    """
    Obtains visibilities from a given configuration
    """ 

    latitude = config["latitude"]
    longitude = config["longitude"]
    file_route = config["file_route"]
    catalog_source = config["catalog_source"]
    utc_start = config["utc_start"]
    utc_end = config["utc_end"]
    step_min = config["step_min"]
    n_freqs = config["n_freqs"]
    interferometer = config["interferometer"]
    n_sources = config["n_sources"]
    max_offset_deg = config["max_offset_deg"]
    flux_range = tuple(config["flux_range"])
    seed = config.get("seed", None)  # por si no está presente

    # 1. Lectura de antenas y definición de parámetros del arreglo
    enu = read_cfg_to_enu(file_route)

    # 2. Obtención de Baselines
    baselines_enu = baselines(enu.T, False)

    # 3. Transformacion baseline -> alt, az (Horizontales) -> XYZ (Ecuatoriales)
    alt, az = enu_to_altaz(baselines_enu.T, rad=True)
    r_eq = hor_to_eq(baselines_enu, alt, az, phi=latitude)

    # 4. Transformación XYZ -> uvw
    sirius_dec = SOURCES_CATALOG[catalog_source]['Dec']
    sirius_ra = SOURCES_CATALOG[catalog_source]['RA']
    delta_src = ra_dec_to_radians(sirius_dec, is_ra=False)
    ra_src = ra_dec_to_radians(sirius_ra)

    # Rango de horas angulo
    times_utc, H, lst = H_range(
        ra_rad=ra_src,
        utc_start=utc_start,
        utc_end=utc_end,
        longitude=longitude,
        step_minutes=step_min
    )

    # Obtención muestreo
    uvw = eq_to_uvw(H, delta_src, r_eq)
    # 6. Transformación a longitud de Onda

    if interferometer["name"] == "VLA":
        frequencies = select_vla_frequencies(interferometer["band_name"], n_freqs)

    uvw_lambda, wavelengths = uvw_to_lambda_range(uvw, frequencies)

    ra0_deg = np.degrees(ra_dec_to_radians(SOURCES_CATALOG['Sirius']['RA']))  # Sirius A
    dec0_deg = np.degrees(ra_dec_to_radians(SOURCES_CATALOG['Sirius']['Dec']))
    sources = generate_random_sources(ra0_deg, dec0_deg, N=n_sources, max_offset_deg=max_offset_deg, flux_range=flux_range, seed=seed)

    V, omega, l_src, m_src, n_src = visibilities_from_sources(uvw_lambda, sources, ra0_deg, dec0_deg)

    return V, uvw_lambda, frequencies, baselines_enu