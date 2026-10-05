"""Figure 4. How steep a ceiling can still hold gas.

Saturated gas states (every inverted depression filled to its spill
point) on 512 x 512 ceilings, 10.24 m square, sigma1 = 0.10 m, no arch
and no pinning, six realizations per dip.  Dips are chosen so that the
closed-form scale r* = (sigma1/tan beta)^(1/(1-H)) runs from 0.1 to
0.8 m.
(a) Saturated states for H = 0.8 at three dips (gas thickness).
(b, c) Fraction of the pooled area held in pools of linear size
sqrt(A) larger than l, against l/r*, for H = 0.5 (b) and H = 0.8 (c).
Curves that coincide collapse on r*.
(d) Fraction of the ceiling holding gas at saturation against dip.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from cave_percolation.anim import GAS, ROCK, hillshade
from cave_percolation.ceiling import build_ceiling
from cave_percolation.fillspill import priority_flood, pool_stats
from cave_percolation.io_utils import write_csv, save_cache, load_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

N, NR = 512, 6
HS = [0.5, 0.8]
RSTAR = np.logspace(-1, np.log10(0.8), 4)
Q = np.logspace(np.log10(0.25), np.log10(16), 25)

cache = load_cache("fig04")
if cache is None:
    out = {}
    for hu in HS:
        tb = PRM.sigma1 / RSTAR ** (1 - hu)
        F, fr = np.zeros((len(tb), len(Q))), np.zeros((len(tb), NR))
        for i, t in enumerate(tb):
            tilt = np.degrees(np.arctan(t))
            As = []
            for k, s in enumerate(PRM.seeds[:NR]):
                rng = np.random.default_rng(s)
                h, he = build_ceiling(PRM, rng, tilt_deg=tilt, hurst=hu,
                                      arch=0.0, nx=N, ny=N, pinning=False)
                th = priority_flood(he, PRM.a, t)
                A, _ = pool_stats(th, PRM.a)
                As.append(A)
                fr[i, k] = (th > 0).mean()
                if hu == 0.8 and k == 0 and i in (0, 2, 3):
                    out[f"map{i}"] = th
                    out[f"he{i}"] = he
            A = np.concatenate(As)
            ls = np.sqrt(A)
            F[i] = [A[ls > q * RSTAR[i]].sum() / A.sum() for q in Q]
        out[f"F{hu}"], out[f"fr{hu}"], out[f"tb{hu}"] = F, fr, tb
    save_cache("fig04", **out)
    cache = load_cache("fig04")

setup()
fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.5,
                      wspace=0.45)
vmax = max(np.percentile(cache[f"map{i}"][cache[f"map{i}"] > 0], 99)
           for i in (0, 2, 3)) * 1e3
ext = [0, N * PRM.a, 0, N * PRM.a]
for j, i in enumerate((0, 2, 3)):
    ax = fig.add_subplot(gs[0, j])
    he, th = cache[f"he{i}"], cache[f"map{i}"]
    ax.imshow(ROCK(0.25 + 0.75 * hillshade(he, PRM.a)), extent=ext,
              origin="lower", interpolation="bilinear")
    im = ax.imshow(np.ma.masked_where(th <= 0, 1e3 * th), cmap=GAS,
                   extent=ext, origin="lower", vmin=0, vmax=vmax,
                   interpolation="nearest")
    tbv = cache["tb0.8"][i]
    ax.set_title(rf"$\beta={np.degrees(np.arctan(tbv)):.1f}^\circ$,"
                 rf" $r^*={RSTAR[i]:.2g}$ m", fontsize=8)
    ax.set_xlabel(r"$x$ (m), up-dip $\rightarrow$")
    if j == 0:
        ax.set_ylabel(r"$y$ (m)")
        panel_label(ax, "(a)", "Saturated pools, $H=0.8$", dy=1.14)
cax = fig.add_axes([0.915, 0.58, 0.012, 0.32])
cb = fig.colorbar(im, cax=cax)
cb.set_label("gas thickness (mm)")

tab = {"l_over_rstar": Q}
bc = []
for j, hu in enumerate(HS):
    ax = fig.add_subplot(gs[1, j])
    bc.append(ax)
    F = cache[f"F{hu}"]
    for i in range(len(RSTAR)):
        ax.loglog(Q, np.where(F[i] > 0, F[i], np.nan), "o-", ms=2.5,
                  color=SEQ[i + 1], mfc="none", lw=0.9)
        tab[f"F_H{hu}_rstar{RSTAR[i]:.3f}"] = F[i]
    ax.set_xlabel(r"$\ell / r^*$")
    ax.set_ylim(3e-3, 1.2)
    if j == 0:
        ax.set_ylabel(r"pooled area in pools $>\ell$")
    panel_label(ax, "(b)" if j == 0 else "(c)",
                rf"Pool-size collapse, $H={hu}$")
write_csv("fig04bc_collapse", tab)

ax = fig.add_subplot(gs[1, 2])
t2 = {}
for hu, mk in zip(HS, ("o", "s")):
    tb, fr = cache[f"tb{hu}"], cache[f"fr{hu}"]
    deg = np.degrees(np.arctan(tb))
    ax.errorbar(deg, 100 * fr.mean(1), yerr=100 * fr.std(1), fmt=mk + "-",
                color="k", ms=3.2, mfc="none", capsize=2, lw=0.9)
    t2[f"dip_deg_H{hu}"] = deg
    t2[f"frac_H{hu}"] = fr.mean(1)
    t2[f"frac_sd_H{hu}"] = fr.std(1)
write_csv("fig04d_fraction", t2)
ax.set_xlabel(r"dip $\beta$ (deg)")
ax.set_ylabel("ceiling under gas (%)")
ax.set_ylim(bottom=0)
panel_label(ax, "(d)", "Ceiling under gas")

fig.subplots_adjust(left=0.08, right=0.9, top=0.93, bottom=0.12)
hh, ll = handles([rf"$r^*={r:.2g}$ m" for r in RSTAR], SEQ[1:5],
                 ["-"] * 4, ["o"] * 4)
legend_below(fig, bc, hh, ll, ncol=4)
hh, ll = handles([r"$H=0.5$", r"$H=0.8$"], ["k", "k"], ["-", "-"],
                 ["o", "s"])
legend_below(fig, ax, hh, ll, ncol=2)
print(save(fig, "fig04_tilt"))
