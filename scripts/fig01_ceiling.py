"""Figure 1. The ceiling and the pools it can hold.

(a) Effective ceiling relief (hillshade) and the gas thickness of the
saturated state, in which every inverted depression is filled to its spill
point, for H = 0.8, sigma1 = 0.10 m and no dip.
(b) Rms height difference S(r) against lag for H = 0.5 and 0.8, eight
realizations each; dashed lines are sigma1 r^H.
(c) Volume against area of every pool in the saturated states of the same
realizations with at least five cells.  Dashed lines have the closed-form
slope (2 + H)/2, the inverse of the area-volume exponent 2/(2 + H).
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from cave_percolation.anim import ROCK, GAS, hillshade
from cave_percolation.ceiling import build_ceiling, structure_function
from cave_percolation.fillspill import priority_flood, pool_stats
from cave_percolation.io_utils import write_csv, save_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

N, NR = 256, 8
HS = [0.5, 0.8]
COLS = [C["blue"], C["vermil"]]
LAGS = np.unique(np.round(np.logspace(0, np.log10(N / 4), 18)).astype(int))

res = {}
for hu in HS:
    S, A, V = [], [], []
    for s in PRM.seeds[:NR]:
        rng = np.random.default_rng(s)
        h, he = build_ceiling(PRM, rng, tilt_deg=0.0, hurst=hu, arch=0.0,
                              nx=N, ny=N, pinning=False)
        r, sf = structure_function(h, PRM.a, LAGS)
        S.append(sf)
        th = priority_flood(he, PRM.a, 0.0)
        ar, vo = pool_stats(th, PRM.a)
        A.append(ar)
        V.append(vo)
        if hu == 0.8 and s == PRM.seeds[0]:
            keep = (he, th)
    res[hu] = (r, np.array(S), np.concatenate(A), np.concatenate(V))

setup()
fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.5)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.2)
ax = fig.add_subplot(gs[0])
he, th = keep
ext = [0, N * PRM.a, 0, N * PRM.a]
shade = hillshade(he, PRM.a)
ax.imshow(ROCK(0.25 + 0.75 * shade), extent=ext, origin="lower",
          interpolation="bilinear")
thm = np.ma.masked_where(th <= 0, 1e3 * th)
im = ax.imshow(thm, cmap=GAS, extent=ext, origin="lower", vmin=0,
               vmax=np.percentile(1e3 * th[th > 0], 99),
               interpolation="nearest")
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel(r"$y$ (m)")
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
cb.set_label("gas thickness (mm)")
panel_label(ax, "(a)", "Saturated gas pools")
save_cache("fig01_map", he=he, th=th)

axb = fig.add_subplot(gs[1])
tab = {}
for hu, col in zip(HS, COLS):
    r, S, _, _ = res[hu]
    m = S.mean(0)
    axb.loglog(r, 100 * m, "o", color=col, ms=3, mfc="none")
    axb.loglog(r, 100 * PRM.sigma1 * r ** hu, "--", color=col, lw=0.9)
    tab["r"] = r
    tab[f"S_H{hu}"] = m
    tab[f"S_H{hu}_sd"] = S.std(0)
axb.set_xlabel(r"lag $r$ (m)")
axb.set_ylabel(r"$S(r)$ (cm)")
panel_label(axb, "(b)", "Relief roughness")
write_csv("fig01b_structure_function", tab)

axc = fig.add_subplot(gs[2])
fits = []
for hu, col in zip(HS, COLS):
    _, _, A, V = res[hu]
    big = A >= 5 * PRM.a ** 2
    axc.loglog(A[big], V[big] * 1e3, ".", color=col, ms=1.5, alpha=0.35,
               rasterized=True)
    aa = np.logspace(np.log10(4 * PRM.a ** 2), np.log10(A.max()), 10)
    sel = A >= 10 * PRM.a ** 2
    pref = np.exp(np.median(np.log(V[sel]) - (1 + hu / 2)
                            * np.log(A[sel])))
    axc.loglog(aa, 1e3 * pref * aa ** (1 + hu / 2), "--", color="k",
               lw=0.9)
    write_csv(f"fig01c_pools_H{hu}", {"area": A, "volume": V})
    la, lv = np.log(A[sel]), np.log(V[sel])
    boot = []
    brng = np.random.default_rng(0)
    for _ in range(1000):
        i = brng.integers(0, la.size, la.size)
        boot.append(np.polyfit(la[i], lv[i], 1)[0])
    fits.append([hu, 1 + hu / 2, np.polyfit(la, lv, 1)[0],
                 *np.percentile(boot, [2.5, 97.5]), la.size])
axc.set_xlabel(r"pool area $A$ (m$^2$)")
axc.set_ylabel(r"pool volume $V$ (L)")
panel_label(axc, "(c)", "Pool volume and area")

fits = np.array(fits)
write_csv("fig01c_fit", {"H": fits[:, 0], "expected": fits[:, 1],
                         "slope": fits[:, 2], "ci_lo": fits[:, 3],
                         "ci_hi": fits[:, 4], "n_pools": fits[:, 5]})
hfit = []
for hu in HS:
    r, S, _, _ = res[hu]
    m = (r >= 0.04) & (r <= 1.0)
    hfit.append(np.polyfit(np.log(r[m]), np.log(S[:, m].mean(0)), 1)[0])
write_csv("fig01b_fit", {"H_input": np.array(HS), "H_fit": np.array(hfit)})
hh, ll = handles([rf"$H={hu}$" for hu in HS] + [r"$\sigma_1 r^H$"],
                 COLS + [C["gray"]], ["-", "-", "--"], ["o", "o", None])
legend_below(fig, axb, hh, ll, ncol=3)
hh, ll = handles([rf"$H={hu}$" for hu in HS] + [r"slope $(2+H)/2$"],
                 COLS + ["k"], ["-", "-", "--"], [".", ".", None])
legend_below(fig, axc, hh, ll, ncol=3)
print(save(fig, "fig01_ceiling"))
