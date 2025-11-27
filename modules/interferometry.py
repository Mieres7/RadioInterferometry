"""
Cálculos de visibilidades, frecuencias y grillas
"""

import numpy as np
from cupy.fft import fftshift, ifftshift, ifft2, fft2
from modules.gridder import grid_visibilities

def uvw_to_lambda(uvw, freq_hz):
    c = 299792458.0
    lam = c / freq_hz
    return uvw / lam, lam


def uvw_to_lambda_range(uvw, freqs_hz):
    c = 299792458.0
    freqs_hz = np.asarray(freqs_hz)
    lams = c / freqs_hz
    uvw_lambda = uvw[:, :, np.newaxis, :] / lams[np.newaxis, np.newaxis, :, np.newaxis]
    return uvw_lambda, lams


def direction_cosines(ra_rad, dec_rad, ra0_rad, dec0_rad):
    dalpha = ra_rad - ra0_rad
    cosd, sind = np.cos(dec_rad), np.sin(dec_rad)
    cosd0, sind0 = np.cos(dec0_rad), np.sin(dec0_rad)
    l = cosd * np.sin(dalpha)
    m = sind * cosd0 - cosd * sind0 * np.cos(dalpha)
    n = sind * sind0 + cosd * cosd0 * np.cos(dalpha)
    return l, m, n


def visibilities_from_sources(uvw_lambda, sources, ra0_deg, dec0_deg, sigma_pb=0.05):
    u, v, w = uvw_lambda[..., 0], uvw_lambda[..., 1], uvw_lambda[..., 2]
    ra0, dec0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)

    ras = np.array([src['ra_deg'] for src in sources])
    decs = np.array([src['dec_deg'] for src in sources])
    S0s = np.array([src.get('S0', 1.0) for src in sources])

    ras_rad, decs_rad = np.deg2rad(ras), np.deg2rad(decs)
    l_src, m_src, n_src = direction_cosines(ras_rad, decs_rad, ra0, dec0)
    A_src = np.exp(-(l_src**2 + m_src**2) / (2 * sigma_pb**2))

    V_total = np.zeros(u.shape, dtype=np.complex64)
    for ls, ms, ns, As, Ss in zip(l_src, m_src, n_src, A_src, S0s):
        phase = 2j * np.pi * (u * ls + v * ms + w * (ns - 1.0))
        V_total += As * Ss / ns * np.exp(phase)

    omega = np.ones_like(V_total, dtype=float)
    return V_total, omega, l_src, m_src, n_src


def generate_random_sources(ra0_deg, dec0_deg, N=50, max_offset_deg=1.0, flux_range=(0.1, 2.0), seed=None):
    rng = np.random.default_rng(seed)
    ras = ra0_deg + rng.uniform(-max_offset_deg, max_offset_deg, N)
    decs = dec0_deg + rng.uniform(-max_offset_deg, max_offset_deg, N)
    fluxes = rng.uniform(flux_range[0], flux_range[1], N)
    return [{"ra_deg": ra, "dec_deg": dec, "S0": S} for ra, dec, S in zip(ras, decs, fluxes)]


def to_image(visibilities):
    return fftshift(ifft2(ifftshift(visibilities)))

def to_fourier(image):
    return fftshift(fft2(ifftshift(image)))

def forward_op(data=None, gridded=False, grid_func='numpy', Image=None):
    
    if data is not None:
        V = data['V']
        uvw = data.get('uvw', None)
        du = data.get('du',None)
        dv = data.get('dv', None)
        Npix = data.get('N', None)

    if not gridded:
        VG, _ = grid_visibilities(V, uvw, du, dv, Npix=Npix, mode=grid_func)
        return to_fourier(VG)
    else:
        # this is with no grid, check!
        return to_fourier(Image)

def adjoint_op(visibilities):
    return to_image(visibilities)

    

