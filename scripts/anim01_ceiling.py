"""Animation 1. Looking up: a team of three passes under a silty ceiling.

The reference passage (12.8 m x 3.2 m, dip 2 deg toward -x, arch 0.15 m)
seen from below, one ceiling realization.  Three exact color scales: rock
color is the loose silt left on each cell (g m^-2, ochre where coated,
gray where stripped), cells crossed by migrating gas show the cumulative
gas volume that has passed (L, logarithmic), and trapped gas shows its
thickness h_eff - L (mm).  Thin contours mark the relief every 4 cm
(dip removed).  Dashed circles mark the 2-sigma bubble footprint above
each diver's regulator.  Divers swim up-dip (+x) at 0.2 m/s, 20 s apart,
RMV 20 L/min.  The scripts check that no value exceeds its color bar.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from cave_percolation.anim import (GAS, TRAIL, TRAIL_NORM, dark, trail_layer,
                                   fig_to_rgb, write_gif, scalebar)
from cave_percolation.experiments import passage
from cave_percolation.divers import transit, run_scenario
from cave_percolation.scenario import PRM

SEED = PRM.seeds[0]
h, he, eng, rng = passage(PRM, SEED)
par = transit(PRM, rng, n_divers=3)
t_end = par[-1][0] + 12.0
T = np.arange(0.0, t_end, 1.5)
silt0 = PRM.m0 * PRM.a ** 2
snaps = []


def grab(e, t):
    """Store the fields needed for one frame."""
    snaps.append((t, e.thickness(), e.silt.reshape(e.ny, e.nx) / silt0,
                  e.flux.reshape(e.ny, e.nx).copy()))


run_scenario(eng, par, T, grab)

CLEAN = np.array([0.52, 0.55, 0.59])
SILTY = np.array([0.58, 0.47, 0.34])
COAT = LinearSegmentedColormap.from_list("coat", [CLEAN, SILTY])
GMAX = 70.0
ext = [0, PRM.length, 0, PRM.width]
yc = 0.5 * PRM.width
xs = (np.arange(PRM.nx) + 0.5) * PRM.a
ys = (np.arange(PRM.ny) + 0.5) * PRM.a
relief = h - np.tan(np.radians(PRM.tilt_deg)) * xs[None, :]
levels = np.arange(np.floor(relief.min() / 0.04) * 0.04, relief.max(), 0.04)
assert max(s[1].max() for s in snaps) * 1e3 <= GMAX
assert max(s[3].max() for s in snaps) * 1e3 <= TRAIL_NORM.vmax

dark()
frames = []
for t, th, sf, fl in snaps:
    img = COAT(sf)[..., :3]
    tcol, tr = trail_layer(fl)
    img = np.where(tr[..., None], tcol, img)
    gas = th > 0
    gcol = GAS(1e3 * th / GMAX)[..., :3]
    img = np.where(gas[..., None], gcol, img)

    fig = plt.figure(figsize=(9.6, 2.9), dpi=90)
    ax = fig.add_axes([0.05, 0.17, 0.72, 0.74])
    ax.imshow(img, extent=ext, origin="lower", interpolation="nearest")
    ax.contour(xs, ys, relief, levels=levels, colors="k", linewidths=0.25,
               alpha=0.35)
    for k in range(3):
        ts = k * PRM.spacing_t
        x = -1.0 + PRM.u_diver * (t - ts)
        if t >= ts and -0.5 < x < PRM.length + 0.5:
            ax.add_patch(plt.Circle((x, yc), 2 * PRM.footprint, fill=False,
                                    ls="--", lw=0.8, color="#7fe3ff"))
            ax.text(x, yc + 2 * PRM.footprint + 0.08, f"D{k + 1}",
                    color="#7fe3ff", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, PRM.length)
    ax.set_ylim(0, PRM.width)
    ax.set_xlabel(r"$x$ along passage (m), up-dip $\rightarrow$")
    ax.set_ylabel(r"$y$ (m)")
    scalebar(ax, 0.3, 0.25, 1.0, "1 m")
    ax.text(0.995, 1.02, f"t = {t:5.0f} s", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    ax.text(0.0, 1.02, "ceiling seen from below", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9)
    cax1 = fig.add_axes([0.79, 0.17, 0.011, 0.74])
    cb = fig.colorbar(ScalarMappable(Normalize(0, GMAX), GAS), cax=cax1)
    cb.set_label("trapped gas (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cax3 = fig.add_axes([0.865, 0.17, 0.011, 0.74])
    cb3 = fig.colorbar(ScalarMappable(TRAIL_NORM, TRAIL), cax=cax3)
    cb3.set_label("gas passed (L)", fontsize=8)
    cb3.ax.tick_params(labelsize=7)
    cax2 = fig.add_axes([0.94, 0.17, 0.011, 0.74])
    cb2 = fig.colorbar(ScalarMappable(Normalize(0, PRM.m0), COAT), cax=cax2)
    cb2.set_label(r"loose silt (g m$^{-2}$)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 16
print(write_gif("anim01_ceiling", frames, fps=10, colors=255))
