"""Scenario runners shared by the figure and animation scripts."""

import numpy as np

from .ceiling import build_ceiling
from .column import fallout, mass_attenuation, beam_attenuation
from .column import sighting_along
from .divers import transit, hover, run_scenario
from .fillspill import FillSpill
from .scenario import D_CLASSES_UM

__all__ = ["passage", "team_run", "hover_run", "visibility_series"]


def passage(prm, seed, tilt_deg=None):
    """Ceiling, effective ceiling, and a silt-loaded engine for one seed."""
    rng = np.random.default_rng(seed)
    tilt = prm.tilt_deg if tilt_deg is None else tilt_deg
    h, he = build_ceiling(prm, rng, tilt_deg=tilt)
    silt = np.full(he.shape, prm.m0 * prm.a ** 2)
    eng = FillSpill(he, prm.a, np.tan(np.radians(tilt)), silt=silt,
                    phi_c=prm.phi_c, v_d=prm.v_d)
    return h, he, eng, rng


def team_run(prm, seed, n_divers=3, rmv=None, tilt_deg=None,
             snapshots=None, callback=None):
    """Run a team transit and return (h, h_eff, engine)."""
    h, he, eng, rng = passage(prm, seed, tilt_deg)
    par = transit(prm, rng, n_divers=n_divers, rmv=rmv)
    run_scenario(eng, par, snapshots, callback)
    return h, he, eng


def hover_run(prm, seed, x_h, duration, rmv=None, tilt_deg=None,
              snapshots=None, callback=None):
    """Run one diver holding station and return (h, h_eff, engine)."""
    h, he, eng, rng = passage(prm, seed, tilt_deg)
    par = hover(prm, rng, x_h, duration, rmv=rmv)
    run_scenario(eng, par, snapshots, callback)
    return h, he, eng


def visibility_series(prm, eng, t_out, seed, kappa=None, n_target=12000,
                      band=0.1, record=None):
    """Settle an engine's releases and return sighting ranges.

    Returns a dict with the output times, the minimum over the passage of
    the sighting range toward the exit (down-dip, -x) in the band
    z_eye +/- ``band`` and in the top 0.1 m under the ceiling, the beam
    attenuation along those two lines, the concentration snapshots, the
    bin edges, and the per-class attenuation.
    """
    t, c, m, _ = eng.releases()
    x = (c % prm.nx + 0.5) * prm.a
    rng = np.random.default_rng(seed + 7919)
    xe = np.linspace(0, prm.length, 129)
    ze = np.linspace(0, prm.h_passage, 41)
    tt, conc, col, xe, ze = fallout(prm, t, x, m, D_CLASSES_UM, t_out, rng,
                                    dt=2.0, n_target=n_target, kappa=kappa,
                                    xe=xe, ze=ze, record=record)
    b = mass_attenuation(prm, D_CLASSES_UM * 1e-6)
    cz = beam_attenuation(prm, conc, b)
    zc = 0.5 * (ze[1:] + ze[:-1])
    eye = np.abs(zc - prm.z_eye) <= band + 1e-9
    top = zc <= 0.1
    dx = xe[1] - xe[0]
    c_eye = cz[:, eye, :].mean(1)
    c_top = cz[:, top, :].mean(1)
    v_eye = sighting_along(prm, c_eye, dx).min(-1)
    v_top = sighting_along(prm, c_top, dx).min(-1)
    return dict(t=tt, v_eye=v_eye, v_top=v_top, conc=conc, xe=xe, ze=ze,
                b=b, c_eye=c_eye, c_top=c_top)
