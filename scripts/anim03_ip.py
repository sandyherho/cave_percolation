"""Animation 3. The connected capillary limit: invasion percolation.

Gas under a smooth flat ceiling enters, one lattice cell at a time, the
perimeter cell with the lowest entry threshold (uniform on (0, 1)), on a
561^2 lattice, for 45 000 cells or until it reaches the edge.  Color is the
order of invasion; the brightest cells are the most recent 1 % of the
invasion, which shows the bursts in which the front jumps through a run of
easy cells.  Water enclosed by gas stays connected to the water below, so
nothing is trapped.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from cave_percolation.anim import BG, dark, fig_to_rgb, write_gif
from cave_percolation.ip import invade

L = 561
rng = np.random.default_rng(2026)
o, acc = invade(rng.random((L, L)), [(L // 2) * L + L // 2],
                max_steps=45000)
n = o.size
ORD = LinearSegmentedColormap.from_list("ord", [
    (0.0, "#1b2a4a"), (0.35, "#3d6d9e"), (0.7, "#8fc1d9"), (1.0, "#e6f4f8")])
steps = np.unique(np.linspace(n / 150, n, 150).astype(int))
img_ord = np.full(L * L, np.nan)
img_ord[o] = np.arange(n)

oy, ox = np.divmod(o, L)
half = 0.5 * max(np.ptp(ox), np.ptp(oy)) + 8
xlo, xhi = ox.mean() - half, ox.mean() + half
ylo, yhi = oy.mean() - half, oy.mean() + half
dark()
frames = []
for k in steps:
    cur = np.where(img_ord < k, img_ord, np.nan).reshape(L, L)
    rgb = np.zeros((L, L, 3))
    rgb[:] = np.array([int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    m = np.isfinite(cur)
    rgb[m] = ORD(cur[m] / n)[:, :3]
    fresh = m & (cur >= k - 0.01 * n)
    rgb[fresh] = np.array([1.0, 0.93, 0.72])
    fig = plt.figure(figsize=(5.2, 4.6), dpi=90)
    ax = fig.add_axes([0.03, 0.05, 0.78, 0.88])
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, 1.015, "invasion percolation, no trapping",
            transform=ax.transAxes, fontsize=8, va="bottom")
    ax.text(1.0, 1.015, f"{k:,} cells", transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right")
    cax = fig.add_axes([0.84, 0.05, 0.03, 0.88])
    cb = fig.colorbar(ScalarMappable(Normalize(0, n / 1e3), ORD), cax=cax)
    cb.set_label(r"invasion step ($10^3$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 20
print(write_gif("anim03_ip", frames, fps=15, colors=200))
