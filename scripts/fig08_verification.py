"""Figure 8. Verification.

(a) Covered area of a paraboloidal dome h = -r^2/(2R), R = 1 m, against
the exact A = 2 sqrt(pi R V), relative error against lattice spacing for
three volumes; mean absolute error over eight random sub-cell center
offsets.  The dashed line has slope 2.
(b) Relative volume residual |V_pools + V_exported - V_injected| /
V_injected after every one of 20 000 random injections on a dipping
rough ceiling with pinning.
(c) Saturated gas thickness from the event-driven fill-spill engine
against the independent priority-flood filling, all cells of twelve
ceilings with dips 0 to 8 deg.
(d) Mean residence time of Lagrangian particles released at the
ceiling against the exact drift-diffusion result, relative error against
Peclet number Pe = w H_p / K for three time steps (20 000 particles; bars
are two standard errors).
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from cave_percolation.ceiling import build_ceiling
from cave_percolation.column import mfpt
from cave_percolation.fillspill import FillSpill, priority_flood
from cave_percolation.io_utils import write_csv, save_cache, load_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

R, LDOM = 1.0, 3.2
AS = np.array([0.08, 0.04, 0.02, 0.01])
VS = [0.01, 0.05, 0.2]
PES = np.logspace(-1, 2, 7)
DTF = [0.02, 0.005, 0.00125]

cache = load_cache("fig08")
if cache is None:
    out = {}
    rng = np.random.default_rng(5)
    err = np.zeros((len(VS), len(AS)))
    for j, a in enumerate(AS):
        n = int(round(LDOM / a))
        for off in rng.random((8, 2)):
            xc = (np.arange(n) + 0.5) * a - 0.5 * LDOM + (off[0] - .5) * a
            yc = (np.arange(n) + 0.5) * a - 0.5 * LDOM + (off[1] - .5) * a
            r2 = xc[None, :] ** 2 + yc[:, None] ** 2
            he = -r2 / (2 * R)
            c0 = int(np.argmax(he))
            eng = FillSpill(he, a, 0.0)
            prev = 0.0
            for i, v in enumerate(VS):
                eng.inject(c0, v - prev)
                prev = v
                A = eng.covered().sum() * a * a
                err[i, j] += abs(A / (2 * np.sqrt(np.pi * R * v)) - 1) / 8
    out["dome_err"] = err

    rng = np.random.default_rng(9)
    h, he = build_ceiling(PRM, rng, tilt_deg=3.0, nx=256, ny=128)
    eng = FillSpill(he, PRM.a, np.tan(np.radians(3.0)))
    res = []
    for k in range(20000):
        eng.inject(int(rng.integers(0, he.size)), 1e-5 * rng.random())
        if k % 20 == 0:
            res.append(abs(eng.stored() + eng.exported - eng.injected)
                       / eng.injected)
    out["cons"] = np.array(res)

    fs_all, pf_all = [], []
    for k, s in enumerate(PRM.seeds):
        tilt = 8.0 * k / (len(PRM.seeds) - 1)
        rng = np.random.default_rng(s)
        h, he = build_ceiling(PRM, rng, tilt_deg=tilt, nx=128, ny=96)
        tb = np.tan(np.radians(tilt))
        eng = FillSpill(he, PRM.a, tb)
        for rep in range(50):
            for c in rng.permutation(he.size):
                eng.inject(int(c), 2e-5)
            if eng.all_full():
                break
        fs_all.append(eng.thickness().ravel())
        pf_all.append(priority_flood(he, PRM.a, tb).ravel())
    out["fs"] = np.concatenate(fs_all)
    out["pf"] = np.concatenate(pf_all)

    H, K = 2.0, 1e-5
    rng = np.random.default_rng(3)
    rel = np.zeros((len(DTF), len(PES)))
    se = np.zeros_like(rel)
    for i, frac in enumerate(DTF):
        for j, pe in enumerate(PES):
            w = pe * K / H
            T = mfpt(H, w, K)
            dt = frac * T
            z = np.zeros(20000)
            t_hit = np.full(z.size, np.nan)
            alive = np.ones(z.size, bool)
            t = 0.0
            s = np.sqrt(2 * K * dt)
            while alive.any() and t < 40 * T:
                t += dt
                idx = np.flatnonzero(alive)
                z[idx] = np.abs(z[idx] + w * dt
                                + s * rng.standard_normal(idx.size))
                hit = idx[z[idx] >= H]
                t_hit[hit] = t
                alive[hit] = False
            rel[i, j] = np.nanmean(t_hit) / T - 1
            se[i, j] = np.nanstd(t_hit) / np.sqrt(np.isfinite(t_hit).sum()
                                                  ) / T
    out["mfpt_rel"], out["mfpt_se"] = rel, se
    save_cache("fig08", **out)
    cache = load_cache("fig08")

setup()
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
ax = axs[0, 0]
err = cache["dome_err"]
for i, (v, col) in enumerate(zip(VS, SEQ[1:])):
    ax.loglog(AS * 100, err[i], "o-", color=col, ms=3.5, mfc="none")
ax.loglog(AS * 100, err[-1, 0] * (AS / AS[0]) ** 2, "--", color=C["gray"],
          lw=0.9)
ax.set_xticks([1, 2, 4, 8])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"lattice spacing $a$ (cm)")
ax.set_ylabel("area, relative error")
panel_label(ax, "(a)", "Dome area convergence")
write_csv("fig08a_dome", {"a_m": AS, **{f"relerr_V{v:g}": err[i]
                                        for i, v in enumerate(VS)}})

ax = axs[0, 1]
cons = np.maximum(cache["cons"], 1e-17)
ax.semilogy(np.arange(cons.size) * 20, cons, color="k", lw=0.6)
ax.set_ylim(1e-17, 1e-10)
ax.set_xlabel("injection event")
ax.set_ylabel("relative volume residual")
panel_label(ax, "(b)", "Volume conservation")
write_csv("fig08b_conservation", {"event": np.arange(cons.size) * 20,
                                  "residual": cache["cons"]})

ax = axs[1, 0]
fs, pf = cache["fs"], cache["pf"]
m = (fs > 0) | (pf > 0)
ax.plot(1e3 * pf[m], 1e3 * fs[m], ".", color="k", ms=1.2, alpha=0.4,
        rasterized=True)
lim = [0, 1e3 * pf.max() * 1.05]
ax.plot(lim, lim, "--", color=C["vermil"], lw=0.9)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("priority flood (mm)")
ax.set_ylabel("fill-spill engine (mm)")
panel_label(ax, "(c)", "Fill-spill vs. priority flood")
write_csv("fig08c_saturation", {"n_cells": [fs.size],
                                "n_gas_cells": [int(m.sum())],
                                "max_abs_diff_m": [np.abs(fs - pf).max()]})

ax = axs[1, 1]
rel, se = cache["mfpt_rel"], cache["mfpt_se"]
DCOL = [C["purple"], C["vermil"], "k"]
for i, (frac, col) in enumerate(zip(DTF, DCOL)):
    ax.errorbar(PES * (1 + 0.04 * (i - 1)), 100 * rel[i], yerr=200 * se[i],
                fmt="s-", color=col, ms=3.2, mfc="none", capsize=2, lw=0.9)
ax.axhline(0, color=C["gray"], lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"Peclet number $wH_p/K$")
ax.set_ylabel("residence time error (%)")
panel_label(ax, "(d)", "Residence time")
tab = {"Pe": PES}
for i, frac in enumerate(DTF):
    tab[f"relerr_dt{frac:g}T"] = rel[i]
    tab[f"se_dt{frac:g}T"] = se[i]
write_csv("fig08d_residence", tab)

fig.tight_layout(h_pad=3.2, w_pad=1.2)
hh, ll = handles([rf"$V={v:g}$ m$^3$" for v in VS] + ["slope 2"],
                 SEQ[1:4] + [C["gray"]], ["-"] * 3 + ["--"],
                 ["o"] * 3 + [None])
legend_below(fig, axs[0, 0], hh, ll, ncol=2)
hh, ll = handles([rf"$\Delta t={f:g}\,T$" for f in DTF],
                 [C["purple"], C["vermil"], "k"], ["-"] * 3, ["s"] * 3)
legend_below(fig, axs[1, 1], hh, ll, ncol=3)
hh, ll = handles(["cells", "1:1"], ["k", C["vermil"]], ["", "--"],
                 [".", None])
legend_below(fig, axs[1, 0], hh, ll, ncol=2)
print(save(fig, "fig08_verification"))
print("dome", err)
print("cons max", cache["cons"].max(), "sat maxdiff", np.abs(fs - pf).max())
print("mfpt", np.round(100 * rel, 3), np.round(200 * se, 3))
