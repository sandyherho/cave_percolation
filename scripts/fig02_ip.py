"""Figure 2. The connected capillary limit reproduces percolation.

(a) One invasion-percolation cluster grown from the center of a 1201^2
lattice until it touches the edge, colored by invasion step.
(b) Number of invaded sites against radius of gyration during growth,
32 realizations (gray) and their mean (black); the dashed line has the
exact slope 91/48.
(c) Density of accepted thresholds, normalized by its plateau; the
dashed line is the square-lattice site threshold p_c = 0.592746.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cave_percolation.io_utils import write_csv, save_cache
from cave_percolation.ip import invade, D_F, P_C
from cave_percolation.plotting import (setup, OKABE_ITO as C, panel_label,
                                       legend_below, handles, save)

L, NR = 1201, 32
c0 = (L // 2) * L + L // 2
kgrid = np.unique(np.logspace(2, 5.2, 36).astype(int))
curves, Ds, pcs = [], [], []
edges = np.linspace(0, 1, 401)
hist = np.zeros(400)
for s in range(NR):
    rng = np.random.default_rng(1000 + s)
    thr = rng.random((L, L))
    o, acc = invade(thr, [c0])
    iy, ix = np.divmod(o, L)
    n = np.arange(1, o.size + 1)
    cx, cy = np.cumsum(ix) / n, np.cumsum(iy) / n
    rg = np.sqrt(np.cumsum(ix ** 2.0) / n - cx ** 2
                 + np.cumsum(iy ** 2.0) / n - cy ** 2)
    k = kgrid[kgrid < o.size] - 1
    curves.append(np.interp(np.log(kgrid), np.log(k + 1), np.log(rg[k]),
                            right=np.nan))
    Ds.append(np.polyfit(np.log(rg[k]), np.log(n[k]), 1)[0])
    h, _ = np.histogram(acc, bins=edges)
    hist += h
    if s == 0:
        keep = o
curves = np.array(curves)
Ds = np.array(Ds)
dens = hist / np.median(hist[edges[:-1] < 0.5])
mid = 0.5 * (edges[1:] + edges[:-1])
i = np.argmax((mid > 0.5) & (dens < 0.5))
pc_est = mid[i - 1] + (0.5 - dens[i - 1]) / (dens[i] - dens[i - 1]) * (
    mid[i] - mid[i - 1])
brng = np.random.default_rng(0)
boot = [Ds[brng.integers(0, NR, NR)].mean() for _ in range(2000)]
write_csv("fig02b_growth", {"n": kgrid, "log_rg_mean":
                            np.nanmean(curves, 0)})
write_csv("fig02b_fit", {"D_mean": [Ds.mean()], "ci_lo":
                         [np.percentile(boot, 2.5)], "ci_hi":
                         [np.percentile(boot, 97.5)], "D_exact": [D_F],
                         "n_real": [NR]})
write_csv("fig02c_acceptance", {"threshold": mid, "density": dens})
write_csv("fig02c_pc", {"pc_estimate": [pc_est], "pc_exact": [P_C]})
save_cache("fig02_cluster", order=keep, L=np.array([L]))

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5),
                        gridspec_kw={"width_ratios": [1, 1, 1]})
ax = axs[0]
img = np.full(L * L, np.nan)
img[keep] = np.arange(1, keep.size + 1)
img = img.reshape(L, L)
im = ax.imshow(img, cmap="viridis", norm=LogNorm(10, keep.size),
               interpolation="nearest", origin="lower", rasterized=True)
ky, kx = np.divmod(keep, L)
half = 0.5 * max(np.ptp(kx), np.ptp(ky)) + 10
ax.set_xlim(kx.mean() - half, kx.mean() + half)
ax.set_ylim(ky.mean() - half, ky.mean() + half)
ax.set_xticks([])
ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
cb.set_label("invasion step")
panel_label(ax, "(a)", "Invasion cluster")

ax = axs[1]
rg_m = np.exp(np.nanmean(curves, 0))
for c in curves:
    ax.loglog(np.exp(c), kgrid, color=C["gray"], lw=0.4, alpha=0.4)
ax.loglog(rg_m, kgrid, color="k", lw=1.3)
ok = np.isfinite(rg_m)
pref = np.exp(np.median(np.log(kgrid[ok]) - D_F * np.log(rg_m[ok])))
rr = np.array([rg_m[ok].min(), rg_m[ok].max()])
ax.loglog(rr, pref * rr ** D_F, "--", color=C["vermil"], lw=1.0)
ax.set_xlabel(r"radius of gyration $R_g$ (sites)")
ax.set_ylabel(r"invaded sites $N$")
panel_label(ax, "(b)", "Mass scaling")

ax = axs[2]
ax.plot(mid, dens, color="k", lw=1.0)
ax.axvline(P_C, color=C["vermil"], ls="--", lw=1.0)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.25)
ax.set_xlabel("entry threshold")
ax.set_ylabel("accepted density / plateau")
panel_label(ax, "(c)", "Accepted thresholds")

fig.tight_layout(w_pad=1.2)
hh, ll = handles(["realizations", "mean", r"slope $91/48$"],
                 [C["gray"], "k", C["vermil"]], ["-", "-", "--"])
legend_below(fig, axs[1], hh, ll, ncol=3)
hh, ll = handles(["accepted density", r"$p_c=0.5927$"],
                 ["k", C["vermil"]], ["-", "--"])
legend_below(fig, axs[2], hh, ll, ncol=2)
print(save(fig, "fig02_ip"))
print(f"D = {Ds.mean():.4f} [{np.percentile(boot, 2.5):.4f},"
      f" {np.percentile(boot, 97.5):.4f}], pc = {pc_est:.4f}")
