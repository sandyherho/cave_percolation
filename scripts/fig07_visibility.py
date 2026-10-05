"""Figure 7. The way out: sighting range after the team has passed.

Released silt settles as 22 Stokes classes (1 to 128 um, log-normal mass
distribution with median 8 um) with eddy diffusivity K = 1e-5 m^2/s in
still water, in the width-averaged reference passage (ceiling to floor
2 m).  Sighting range is the distance toward the exit (down-dip) at which
the beam optical depth reaches 4.8, clear water 30 m.
(a) Suspended silt 30 min after a team of three (RMV 20) entered, one
ceiling; logarithmic color scale.  Dashed lines bound the eye band,
0.6 to 0.8 m below the ceiling.
(b) Worst sighting range along the passage in the eye band against time
for three scenarios, median and interquartile range of six ceilings; the
thin lines are the top 0.1 m under the ceiling for the team of three.
(c) Worst eye-band sighting range 30 and 60 min after entry against the
loose silt inventory m0, which scales the beam attenuation exactly
(team of three, RMV 20), for K = 1e-6, 1e-5 and 1e-4 m^2/s.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cave_percolation.column import sighting_along
from cave_percolation.experiments import team_run, hover_run, visibility_series
from cave_percolation.io_utils import write_csv, save_cache, load_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

NR = 6
T_OUT = np.arange(60.0, 90 * 60 + 1, 120.0)
SCEN = [("team3", dict(n_divers=3, rmv=20.0)),
        ("team3_rmv40", dict(n_divers=3, rmv=40.0)),
        ("hover5", None)]
KAPPAS = [1e-6, 1e-5, 1e-4]
M0S = np.logspace(0, np.log10(500), 30)

cache = load_cache("fig07")
if cache is None:
    out = {}
    for name, kw in SCEN:
        ve, vt, ce = [], [], []
        for k, s in enumerate(PRM.seeds[:NR]):
            if kw is None:
                _, _, eng = hover_run(PRM, s, 6.4, 300.0)
            else:
                _, _, eng = team_run(PRM, s, **kw)
            r = visibility_series(PRM, eng, T_OUT, s)
            ve.append(r["v_eye"])
            vt.append(r["v_top"])
            ce.append(r["c_eye"])
            if name == "team3" and k == 0:
                i30 = np.argmin(np.abs(T_OUT - 1800))
                out["snap"] = r["conc"][i30].sum(0)
                out["xe"], out["ze"] = r["xe"], r["ze"]
        out[f"ve_{name}"] = np.array(ve)
        out[f"vt_{name}"] = np.array(vt)
        out[f"ce_{name}"] = np.array(ce)
    for kap in KAPPAS:
        ce = []
        for s in PRM.seeds[:4]:
            _, _, eng = team_run(PRM, s, n_divers=3, rmv=20.0)
            r = visibility_series(PRM, eng, T_OUT, s, kappa=kap)
            ce.append(r["c_eye"])
        out[f"ck_{kap:g}"] = np.array(ce)
    save_cache("fig07", **out)
    cache = load_cache("fig07")

setup()
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0], hspace=0.55,
                      wspace=0.32)
ax = fig.add_subplot(gs[0, :])
xe, ze, snap = cache["xe"], cache["ze"], cache["snap"]
im = ax.pcolormesh(xe, ze, np.clip(snap, 0.1, None),
                   norm=LogNorm(0.1, 100), cmap="cividis", shading="flat",
                   rasterized=True)
for z in (PRM.z_eye - 0.1, PRM.z_eye + 0.1):
    ax.axhline(z, color="w", ls="--", lw=0.7)
ax.set_ylim(PRM.h_passage, 0)
ax.set_xlim(0, PRM.length)
ax.set_xlabel(r"$x$ along passage (m), up-dip $\rightarrow$")
ax.set_ylabel("depth below ceiling (m)")
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
cb.set_label(r"silt (mg L$^{-1}$)")
panel_label(ax, "(a)", "Suspended silt 30 min after entry, team of three")

axb7 = ax = fig.add_subplot(gs[1, 0])
cols = {"team3": "k", "team3_rmv40": C["vermil"], "hover5": C["blue"]}
tab = {"t_min": T_OUT / 60}
for name, _ in SCEN:
    v = cache[f"ve_{name}"]
    q1, md, q3 = np.percentile(v, [25, 50, 75], axis=0)
    ax.fill_between(T_OUT / 60, q1, q3, color=cols[name], alpha=0.18, lw=0)
    ax.plot(T_OUT / 60, md, color=cols[name])
    tab.update({f"eye_median_{name}": md, f"eye_q1_{name}": q1,
                f"eye_q3_{name}": q3,
                f"top_median_{name}": np.median(cache[f"vt_{name}"], 0)})
ax.plot(T_OUT / 60, np.median(cache["vt_team3"], 0), color="k", lw=0.6,
        ls=":")
ax.axhline(PRM.vis_clear, color=C["gray"], lw=0.6)
ax.set_yscale("log")
ax.set_xlim(0, 90)
ax.set_xlabel("time since the team entered (min)")
ax.set_ylabel("sighting range toward exit (m)")
panel_label(ax, "(b)", "Sighting range toward the exit")
write_csv("fig07b_sighting", tab)

axc7 = ax = fig.add_subplot(gs[1, 1])
c_w = 4.8 / PRM.vis_clear
dx = xe[1] - xe[0]
tabc = {"m0": M0S}
for kap, ls in zip(KAPPAS, (":", "-", "--")):
    ce = cache[f"ck_{kap:g}"]
    for tt, col in ((1800, C["green"]), (3600, C["purple"])):
        i = np.argmin(np.abs(T_OUT - tt))
        vv = []
        for m0 in M0S:
            line = c_w + (m0 / PRM.m0) * (ce[:, i, :] - c_w)
            vv.append(np.median(sighting_along(PRM, line, dx).min(-1)))
        ax.loglog(M0S, vv, color=col, ls=ls, lw=1.0)
        tabc[f"eye_t{tt // 60}min_K{kap:g}"] = np.array(vv)
ax.axvline(PRM.m0, color=C["gray"], lw=0.6)
ax.set_xlabel(r"loose silt on ceiling $m_0$ (g m$^{-2}$)")
ax.set_ylabel("sighting range toward exit (m)")
panel_label(ax, "(c)", "Sensitivity to $m_0$ and $K$")
write_csv("fig07c_inventory", tabc)

fig.subplots_adjust(left=0.08, right=0.93, top=0.95, bottom=0.12)
hh, ll = handles(["team of 3", "team of 3, RMV 40", "hover 5 min",
                  "top 0.1 m, team of 3"],
                 ["k", C["vermil"], C["blue"], "k"], ["-", "-", "-", ":"])
legend_below(fig, axb7, hh, ll, ncol=2)
hh, ll = handles(["30 min", "60 min", r"$K=10^{-6}$", r"$K=10^{-5}$",
                  r"$K=10^{-4}$ m$^2$ s$^{-1}$"],
                 [C["green"], C["purple"], C["gray"], C["gray"],
                  C["gray"]], ["-", "-", ":", "-", "--"])
legend_below(fig, axc7, hh, ll, ncol=3)
print(save(fig, "fig07_visibility"))
for name, _ in SCEN:
    v = np.median(cache[f"ve_{name}"], 0)
    print(name, [f"{T_OUT[i] / 60:.0f}:{v[i]:.1f}"
                 for i in range(0, len(T_OUT), 5)])
print("top", np.median(cache["vt_team3"], 0)[:6])
