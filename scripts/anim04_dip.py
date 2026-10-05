"""Animation 4. One diver holds station under four ceilings.

The same rough ceiling (sigma1 = 0.10 m, H = 0.8, arch 0.15 m) tilted to
dips of 0, 3, 6 and 10 deg, seen from below, while one open-circuit
diver (RMV 20 L/min) holds station at x = 4 m (dashed circle, 2-sigma
bubble footprint) for 10 min.  Trapped gas thickness (mm) and the
cumulative gas volume that has crossed each cell (L, logarithmic) are
drawn on exact color scales over the hillshaded rock, which carries no
quantity.  Labels give the exhaled gas that has
left the 12.8 m section, in liters and as a percentage of the gas
exhaled so far.  The ceiling is the realization closest to the median
of twelve; on the flat and 3 deg ceilings the pools hold all the gas
for 10 min, as in most realizations, while at 6 and 10 deg it runs
up-dip and leaves the section.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

from cave_percolation.anim import (GAS, ROCK, TRAIL, dark,
                                   hillshade, trail_layer, fig_to_rgb,
                                   write_gif)
from cave_percolation.experiments import hover_run
from cave_percolation.scenario import PRM

TILTS = [0.0, 3.0, 6.0, 10.0]
# Ceiling realization whose exported fractions (0, 0, 56, 91% after
# 10 min) are closest to the twelve-ceiling medians (0, 0, 47, 94%).
SEED = PRM.seeds[7]
X_H, DUR = 4.0, 600.0
T = np.arange(7.5, DUR + 1, 7.5)
GMAX = 210.0
TNORM = LogNorm(1e-2, 300.0)
runs = []
for tl in TILTS:
    snaps = []

    def grab(e, t, snaps=snaps):
        """Store gas thickness and swept cells for one frame."""
        snaps.append((e.thickness(), e.flux.reshape(e.ny, e.nx).copy(),
                      e.exported / max(e.injected, 1e-30), e.exported))
    h, he, eng = hover_run(PRM, SEED, X_H, DUR, tilt_deg=tl,
                           snapshots=T, callback=grab)
    shade = 0.35 + 0.65 * hillshade(h - np.tan(np.radians(tl))
                                    * (np.arange(PRM.nx) + 0.5)[None, :]
                                    * PRM.a, PRM.a, exag=1.5)
    runs.append((shade, snaps))

ext = [0, PRM.length, 0, PRM.width]
assert max(sn[0].max() for _, snaps in runs for sn in snaps) * 1e3 <= GMAX
assert (max(sn[1].max() for _, snaps in runs for sn in snaps) * 1e3
        <= TNORM.vmax)
dark()
frames = []
for i, t in enumerate(T):
    fig = plt.figure(figsize=(8.6, 5.4), dpi=80)
    for j, (tl, (shade, snaps)) in enumerate(zip(TILTS, runs)):
        th, fl, lost, vout = snaps[i]
        img = ROCK(0.2 + 0.7 * shade)[..., :3]
        tcol, tr = trail_layer(fl, TNORM)
        img = np.where(tr[..., None], tcol, img)
        gas = th > 0
        img = np.where(gas[..., None], GAS(1e3 * th / GMAX)[..., :3], img)
        ax = fig.add_axes([0.07, 0.745 - 0.215 * j, 0.62, 0.185])
        ax.imshow(img, extent=ext, origin="lower", interpolation="nearest")
        ax.add_patch(plt.Circle((X_H, 0.5 * PRM.width), 2 * PRM.footprint,
                                fill=False, ls="--", lw=0.8,
                                color="#7fe3ff"))
        ax.set_xlim(0, PRM.length)
        ax.set_ylim(0, PRM.width)
        ax.set_yticks([0, 3])
        ax.text(1.005, 0.5, rf"$\beta={tl:g}^\circ$" + "\n"
                + f"out {1e3 * vout:4.0f} L" + "\n"
                + f"({100 * lost:.0f}%)", transform=ax.transAxes,
                va="center", ha="left", fontsize=8)
        if j < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$x$ along passage (m), up-dip $\rightarrow$")
        if j == 0:
            mm, ss = divmod(int(t), 60)
            ax.text(0.0, 1.05, "one diver holding station, ceiling seen"
                    " from below", transform=ax.transAxes, fontsize=9)
            ax.text(1.0, 1.05, f"t = {mm:02d}:{ss:02d}",
                    transform=ax.transAxes, ha="right", fontsize=9)
    fig.text(0.02, 0.5, r"$y$ (m)", rotation=90, va="center")
    cax = fig.add_axes([0.80, 0.10, 0.013, 0.83])
    cb = fig.colorbar(ScalarMappable(Normalize(0, GMAX), GAS), cax=cax)
    cb.set_label("trapped gas (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cax2 = fig.add_axes([0.905, 0.10, 0.013, 0.83])
    cb2 = fig.colorbar(ScalarMappable(TNORM, TRAIL), cax=cax2)
    cb2.set_label("gas passed (L)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 16
print(write_gif("anim04_dip", frames, fps=10, colors=255))
