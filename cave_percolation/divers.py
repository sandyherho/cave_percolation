"""Divers as moving sources of exhaled gas.

An open-circuit diver exhales a volume RMV T_b / 60 at ambient pressure
once per breathing period T_b; the gas expands by Boyle's law between the
regulator and the ceiling.  Bubbles reach the ceiling in a footprint of
rms radius sigma_f around the point above the regulator, and each breath
is split into equal parcels released over the exhalation.  A closed-circuit
rebreather exhales no gas in level swimming and is represented by RMV = 0.
"""

import numpy as np

__all__ = ["breaths", "transit", "hover", "run_scenario"]


def breaths(prm, pos, t0, t1, rng, rmv=None, tag=0):
    """Gas parcels (t, cell, dv, tag) from one diver between t0 and t1.

    ``pos(t)`` returns the (x, y) of the regulator in meters.  Parcels that
    land outside the modeled passage length are dropped.
    """
    rmv = prm.rmv_lpm if rmv is None else rmv
    if rmv <= 0:
        return []
    scale = rmv / prm.rmv_lpm
    dv = prm.v_breath * scale / prm.n_sub
    out = []
    tb = t0 + rng.uniform(0, prm.t_breath)
    while tb < t1:
        ts = tb + prm.t_exhale * rng.random(prm.n_sub)
        for t in np.sort(ts):
            x, y = pos(t)
            x += prm.footprint * rng.standard_normal()
            y += prm.footprint * rng.standard_normal()
            ix = int(np.floor(x / prm.a))
            iy = int(np.clip(np.floor(y / prm.a), 0, prm.ny - 1))
            if 0 <= ix < prm.nx:
                out.append((float(t), iy * prm.nx + ix, dv, tag))
        tb += prm.t_breath
    return out


def transit(prm, rng, n_divers=3, rmv=None, x0=-1.0, x1=None, t_start=0.0):
    """Parcels for a team swimming up-dip along the passage centerline."""
    x1 = prm.length + 1.0 if x1 is None else x1
    yc = 0.5 * prm.width
    out = []
    for k in range(n_divers):
        ts = t_start + k * prm.spacing_t
        te = ts + (x1 - x0) / prm.u_diver

        def pos(t, ts=ts):
            return x0 + prm.u_diver * (t - ts), yc
        out += breaths(prm, pos, ts, te, rng, rmv=rmv, tag=k)
    out.sort(key=lambda p: p[0])
    return out


def hover(prm, rng, x_h, duration, rmv=None, t_start=0.0, tag=0):
    """Parcels for one diver holding station at ``x_h`` on the centerline."""
    yc = 0.5 * prm.width
    out = breaths(prm, lambda t: (x_h, yc), t_start, t_start + duration,
                  rng, rmv=rmv, tag=tag)
    out.sort(key=lambda p: p[0])
    return out


def run_scenario(engine, parcels, snapshots=None, callback=None):
    """Feed parcels to a FillSpill engine in time order.

    ``callback(engine, t)`` is called at each time in ``snapshots``.
    """
    snaps = [] if snapshots is None else list(snapshots)
    si = 0
    for t, c, dv, tag in parcels:
        while si < len(snaps) and snaps[si] <= t:
            callback(engine, snaps[si])
            si += 1
        engine.inject(c, dv, t, tag)
    while si < len(snaps):
        callback(engine, snaps[si])
        si += 1
    return engine
