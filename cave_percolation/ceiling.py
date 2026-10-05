"""Rough cave ceilings and the closed forms that follow from their scaling.

The ceiling elevation h(x, y) (positive up) is the sum of a self-affine
Gaussian relief, a uniform dip along the passage, and an optional
cross-passage arch.  Gas under the ceiling is excluded from a cell until
its thickness exceeds a pinning height ell = 2 gamma/(rho g r), where r is
a sub-grid contact-line pinning radius, so the gas sees the effective
ceiling h_eff = h - ell.

A self-affine relief has an rms height difference S(r) = sigma1 r^H.  A
dip adds a rise r tan(beta).  The two are equal at

    r* = (sigma1 / tan(beta))^(1/(1 - H)),

the pool cutoff: closed inverted depressions larger than about r* cannot
exist because the dip over their width exceeds their relief.  Below r*
a pool of width l has depth of order l^H and volume of order l^(2+H), so
area grows with volume as A ~ V^(2/(2+H)).
"""

import numpy as np

__all__ = ["self_affine", "structure_function", "build_ceiling",
           "pool_cutoff", "area_volume_exponent", "pinning_height"]


def self_affine(nx, ny, a, hurst, sigma1, rng, pad=4):
    """Isotropic self-affine Gaussian relief on an (ny, nx) lattice.

    Spectral synthesis with power spectrum P(k) ~ k^-(2 + 2H) on a lattice
    ``pad`` times larger in each direction, cropped to (ny, nx).  Cropping
    removes the periodicity of the FFT field, which otherwise flattens the
    structure function at lags approaching the domain size.  The field is
    rescaled so that the rms height difference at a lag of one meter,
    averaged over the two lattice directions, equals ``sigma1``.
    """
    mx, my = pad * nx, pad * ny
    kx = 2 * np.pi * np.fft.fftfreq(mx, a)
    ky = 2 * np.pi * np.fft.fftfreq(my, a)
    k = np.hypot(*np.meshgrid(kx, ky))
    k[0, 0] = 1.0
    amp = k ** (-(1.0 + hurst))
    amp[0, 0] = 0.0
    z = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
    h = np.real(np.fft.ifft2(amp * z))[:ny, :nx]
    lag = int(round(1.0 / a))
    m = min(lag, nx // 2, ny // 2)
    s = structure_function(h, a, [m])[1][0] * (lag / m) ** hurst
    h = h * (sigma1 / s)
    return h - h.mean()


def structure_function(h, a, lags):
    """Rms height difference at integer lattice lags, both directions.

    Returns lag distances (m) and S (m), from non-periodic differences.
    """
    out = []
    for m in lags:
        dx = h[:, m:] - h[:, :-m]
        dy = h[m:, :] - h[:-m, :]
        out.append(np.sqrt(0.5 * (np.mean(dx ** 2) + np.mean(dy ** 2))))
    return np.asarray(lags) * a, np.array(out)


def pinning_height(prm, shape, rng):
    """Sub-grid pinning height ell = 2 gamma/(rho g r), r log-normal."""
    r = prm.r_med * np.exp(prm.r_lnsd * rng.standard_normal(shape))
    return 2.0 * prm.gamma / (prm.rho * prm.g * r)


def build_ceiling(prm, rng, tilt_deg=None, sigma1=None, hurst=None,
                  arch=None, nx=None, ny=None, pinning=True):
    """Return (h, h_eff) on an (ny, nx) lattice; +x is up-dip.

    ``h`` is the geometric ceiling elevation, ``h_eff = h - ell`` the
    effective ceiling seen by an advancing contact line.
    """
    nx = prm.nx if nx is None else nx
    ny = prm.ny if ny is None else ny
    tilt = prm.tilt_deg if tilt_deg is None else tilt_deg
    s1 = prm.sigma1 if sigma1 is None else sigma1
    hu = prm.hurst if hurst is None else hurst
    ar = prm.arch if arch is None else arch
    h = self_affine(nx, ny, prm.a, hu, s1, rng)
    x = (np.arange(nx) + 0.5) * prm.a
    y = (np.arange(ny) + 0.5) * prm.a
    h = h + np.tan(np.radians(tilt)) * x[None, :]
    if ar:
        h = h - ar * (2.0 * y[:, None] / (ny * prm.a) - 1.0) ** 2
    ell = pinning_height(prm, h.shape, rng) if pinning else 0.0 * h
    return h, h - ell


def pool_cutoff(sigma1, hurst, tanb):
    """Closed-form pool cutoff r* = (sigma1/tan beta)^(1/(1-H)) in meters."""
    return (sigma1 / np.asarray(tanb, float)) ** (1.0 / (1.0 - hurst))


def area_volume_exponent(hurst):
    """Exponent of A ~ V^(2/(2+H)) for pools smaller than r*."""
    return 2.0 / (2.0 + hurst)
