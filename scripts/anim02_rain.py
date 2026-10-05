"""Animation 2. Side view: the silt rain after a team of three.

Width-averaged section of the reference passage, ceiling at the top,
for the first hour after a team of three (RMV 20 L/min) entered, one
ceiling realization.  Color is suspended silt from 22 Stokes classes
(logarithmic, 0.1 to 300 mg/L; lower values and clear water take the
bottom color), binned at 0.1 m x 0.05 m; dots are one in 23 of the
model's particles, drawn larger for coarser grains.  The dashed band is the
diver eye band.  The strip below is the sighting range toward the exit
(down-dip, -x) from each point of the eye band.  Open-water diver
silhouettes, drawn to scale along the passage with the second stage at
the regulator depth, show the team while it is in the section.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cave_percolation.anim import SILT, VIS, FG, dark, fig_to_rgb, write_gif
from cave_percolation.column import sighting_along
from cave_percolation.experiments import team_run, visibility_series
from cave_percolation.scenario import PRM, D_CLASSES_UM
from cave_percolation.sketch import aspect_ratio, draw_diver

SEED = PRM.seeds[0]
T = np.concatenate([np.arange(2.0, 180.0, 3.0),
                    np.arange(180.0, 3600.0 + 1, 30.0)])
_, _, eng = team_run(PRM, SEED, n_divers=3)
grains = []


def keep_grains(col, t):
    """Store a fixed subset of particles (by identity) for display."""
    s = col.ids % 23 == 0
    grains.append((col.x[s].copy(), col.z[s].copy(), col.k[s].copy()))


r = visibility_series(PRM, eng, T, SEED, record=keep_grains)
conc = r["conc"].sum(1)
assert conc.max() <= 300.0
xe, ze = r["xe"], r["ze"]
dx = xe[1] - xe[0]
vis = sighting_along(PRM, r["c_eye"], dx)
assert 1.0 <= vis.min() and vis.max() <= PRM.vis_clear + 1e-9

dark()
frames = []
for i, t in enumerate(r["t"]):
    fig = plt.figure(figsize=(9.0, 3.6), dpi=90)
    ax = fig.add_axes([0.07, 0.36, 0.78, 0.56])
    im = ax.imshow(np.clip(conc[i], 1e-2, None), cmap=SILT,
                   norm=LogNorm(0.1, 300), extent=[0, PRM.length,
                                                   PRM.h_passage, 0],
                   aspect="auto", interpolation="bilinear")
    ax.set_xlim(0, PRM.length)
    ax.set_ylim(PRM.h_passage, 0)
    kz = aspect_ratio(ax)
    gx, gz, gk = grains[i]
    ax.scatter(gx, gz, s=0.4 + 5.0 * (D_CLASSES_UM[gk] / 128.0),
               c="#f6e7c4", alpha=0.55, linewidths=0, rasterized=True)
    for z in (PRM.z_eye - 0.1, PRM.z_eye + 0.1):
        ax.axhline(z, color="#7fe3ff", ls="--", lw=0.6, alpha=0.7)
    for k in range(3):
        ts = k * PRM.spacing_t
        x = -1.0 + PRM.u_diver * (t - ts)
        if t >= ts and 0 < x < PRM.length:
            draw_diver(ax, x, PRM.z_reg, scale=1.0, kz=kz, palette="dark",
                       zorder=6)
            ax.text(x - 0.75, PRM.z_reg + 0.36, f"D{k + 1}", color="#9fb4c8",
                    ha="center", fontsize=7)
    ax.axhline(0, color="#a89a86", lw=2.5)
    ax.set_xlim(0, PRM.length)
    ax.set_ylim(PRM.h_passage, 0)
    ax.set_xticklabels([])
    ax.set_ylabel("below ceiling (m)")
    mm, ss = divmod(int(round(t)), 60)
    ax.text(0.995, 1.02, f"t = {mm:02d}:{ss:02d}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    ax.text(0.0, 1.02, "silt falling out after a team of three",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    cax = fig.add_axes([0.87, 0.36, 0.012, 0.56])
    cb = fig.colorbar(im, cax=cax, extend="min")
    cb.set_label(r"silt (mg L$^{-1}$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax2 = fig.add_axes([0.07, 0.16, 0.78, 0.12])
    im2 = ax2.imshow(vis[i][None, :], cmap=VIS, norm=LogNorm(1, 30),
                     extent=[0, PRM.length, 0, 1], aspect="auto",
                     interpolation="bilinear")
    ax2.set_yticks([])
    ax2.set_xlim(0, PRM.length)
    ax2.set_xlabel(r"$x$ along passage (m), up-dip $\rightarrow$")
    ax2.text(-0.01, 0.5, "eye\nband", transform=ax2.transAxes, ha="right",
             va="center", fontsize=7)
    cax2 = fig.add_axes([0.87, 0.16, 0.012, 0.12])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_ticks([1, 3, 10, 30])
    cb2.set_ticklabels(["1", "3", "10", "30"])
    cb2.ax.tick_params(labelsize=7)
    cb2.set_label("sight (m)", fontsize=7)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 16
print(write_gif("anim02_rain", frames, fps=12, colors=255))
