"""Figure 3. A dipping smooth ceiling: gas climbs in one finger.

Invasion percolation with thresholds u - G y, u uniform on (0, 1), from
one seed on the down-dip edge of an 800-site-wide strip until the finger
reaches the up-dip edge.  G is the entry-threshold drop per lattice
step, G = a tan(beta) / Delta_ell for a pinning-height range Delta_ell.
(a) Fingers for three gradients (same horizontal scale).
(b) Rms lateral width of the finger against G, mean of eight
realizations per gradient with 95% bootstrap intervals; the dashed line
has the exact slope -nu/(1 + nu) = -4/7.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from cave_percolation.io_utils import write_csv
from cave_percolation.ip import invade, row_width, NU
from cave_percolation.plotting import (setup, OKABE_ITO as C, panel_label,
                                       legend_below, handles, save)

NX, NY, NR = 800, 1600, 8
GS = np.logspace(-3, -1.5, 7)
SHOW = [GS[0], GS[3], GS[6]]
W, lo, hi, shows = [], [], [], {}
brng = np.random.default_rng(0)
for G in GS:
    ws = []
    for s in range(NR):
        rng = np.random.default_rng(int(1e6 * G) + s)
        thr = rng.random((NY, NX)) - G * np.arange(NY)[:, None]
        o, _ = invade(thr, [NX // 2], stop_edge=False, stop_row=NY - 1)
        rows = np.arange(NY // 4, NY - 40)
        ws.append(np.nanmean(row_width(o, NX, rows)))
        if s == 0 and G in SHOW:
            shows[G] = o
    ws = np.array(ws)
    boot = [ws[brng.integers(0, NR, NR)].mean() for _ in range(2000)]
    W.append(ws.mean())
    lo.append(np.percentile(boot, 2.5))
    hi.append(np.percentile(boot, 97.5))
W, lo, hi = map(np.array, (W, lo, hi))
slope = np.polyfit(np.log(GS), np.log(W), 1)[0]
bs = []
for _ in range(2000):
    e = np.exp(brng.normal(0, 1, W.size)
               * (np.log(hi) - np.log(lo)) / 3.92)
    bs.append(np.polyfit(np.log(GS), np.log(W * e), 1)[0])
write_csv("fig03b_width", {"G": GS, "width": W, "ci_lo": lo, "ci_hi": hi})
write_csv("fig03b_fit", {"slope": [slope], "ci_lo": [np.percentile(bs, 2.5)],
                         "ci_hi": [np.percentile(bs, 97.5)],
                         "exact": [-NU / (1 + NU)]})

setup()
fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 0.55, 0.55, 1.6],
                      wspace=0.12)
for k, G in enumerate(SHOW):
    ax = fig.add_subplot(gs[k])
    iy, ix = np.divmod(shows[G], NX)
    img = np.zeros((NY, NX))
    img[iy, ix] = 1
    xc = int(np.median(ix))
    ax.imshow(img[:, max(0, xc - 150):xc + 150], cmap="Greys", origin="lower",
              aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(rf"$G={G:.3g}$", fontsize=8)
    if k == 0:
        ax.set_ylabel(r"up-dip $\rightarrow$")
        panel_label(ax, "(a)", "Gas fingers on a dipping smooth ceiling")
ax = fig.add_subplot(gs[3])
ax.errorbar(GS, W, yerr=[W - lo, hi - W], fmt="o", color="k", ms=3.5,
            mfc="none", capsize=2, lw=0.9)
pref = np.exp(np.mean(np.log(W) + NU / (1 + NU) * np.log(GS)))
ax.loglog(GS, pref * GS ** (-NU / (1 + NU)), "--", color=C["vermil"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"threshold gradient $G$ (per site)")
ax.set_ylabel("finger width (sites)")
ax.set_yticks([2, 3, 4, 6, 10, 15])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
panel_label(ax, "(b)", "Finger width")
fig.subplots_adjust(left=0.04, right=0.91, top=0.92, bottom=0.18)
hh, ll = handles(["mean, 95% interval", r"slope $-4/7$"],
                 ["k", C["vermil"]], ["-", "--"], ["o", None])
legend_below(fig, ax, hh, ll, ncol=2)
print(save(fig, "fig03_gradient"))
print(f"slope {slope:.4f} [{np.percentile(bs, 2.5):.4f},"
      f" {np.percentile(bs, 97.5):.4f}] vs {-NU / (1 + NU):.4f}")
