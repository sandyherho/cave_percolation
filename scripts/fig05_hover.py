"""Figure 5. Gas from one place: the hovering diver.

(a) Covered area against injected volume for gas released at one point under
ceilings with no dip, arch or pinning (512 x 512, 10.24 m square), eight
realizations per Hurst exponent.  Dashed lines have the closed-form slope
2/(2 + H); runs stop when gas first reaches an open end.
(b) Silt released within the modeled 12.8 m section by one open-circuit
diver (RMV 20 L/min) holding station for 10 min in the reference passage
(arch 0.15 m, pinning on), at four dips; median and interquartile range of
twelve ceilings.
(c) Where that silt comes from: mass-weighted up-dip distance of the
released silt from the diver, truncated at the section end, (left axis,
circles) and the fraction of the exhaled gas that leaves the 12.8 m section
(right axis, squares).
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from cave_percolation.ceiling import build_ceiling, area_volume_exponent
from cave_percolation.experiments import hover_run
from cave_percolation.fillspill import FillSpill
from cave_percolation.io_utils import write_csv, save_cache, load_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

N, NR = 512, 8
HS = [0.5, 0.8]
VS = np.logspace(-5, 0, 31)
TILTS = [0.0, 3.0, 6.0, 10.0]
X_H, DUR = 4.0, 600.0
TS = np.linspace(0, DUR, 61)

cache = load_cache("fig05")
if cache is None:
    out = {}
    for hu in HS:
        A = np.full((NR, VS.size), np.nan)
        for k, s in enumerate(PRM.seeds[:NR]):
            rng = np.random.default_rng(s)
            h, he = build_ceiling(PRM, rng, tilt_deg=0.0, hurst=hu,
                                  arch=0.0, nx=N, ny=N, pinning=False)
            eng = FillSpill(he, PRM.a, 0.0)
            c0 = (N // 2) * N + N // 2
            prev = 0.0
            for i, v in enumerate(VS):
                eng.inject(c0, v - prev)
                prev = v
                if eng.exported > 0:
                    break
                A[k, i] = eng.covered().sum() * PRM.a ** 2
        out[f"A{hu}"] = A
    for tl in TILTS:
        S = np.zeros((len(PRM.seeds), TS.size))
        dist, lost = [], []
        for k, s in enumerate(PRM.seeds):
            h, he, eng = hover_run(PRM, s, X_H, DUR, tilt_deg=tl)
            t, c, m, _ = eng.releases()
            S[k] = [m[t <= tt].sum() for tt in TS]
            x = (c % PRM.nx + 0.5) * PRM.a
            dist.append(np.sum(m * (x - X_H)) / m.sum())
            lost.append(eng.exported / eng.injected)
        out[f"S{tl}"] = S
        out[f"dist{tl}"] = np.array(dist)
        out[f"lost{tl}"] = np.array(lost)
    save_cache("fig05", **out)
    cache = load_cache("fig05")

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axs[0]
fits = []
for hu, col, mk in zip(HS, ("k", C["gray"]), ("o", "s")):
    A = cache[f"A{hu}"]
    m = np.exp(np.nanmean(np.log(A), 0))
    ok = np.isfinite(A).all(0)
    ax.loglog(1e3 * VS[ok], m[ok], mk, color=col, ms=3, mfc="none")
    ex = area_volume_exponent(hu)
    fit_rng = ok & (m > 25 * PRM.a ** 2)
    pref = np.exp(np.mean(np.log(m[fit_rng]) - ex * np.log(VS[fit_rng])))
    ax.loglog(1e3 * VS[ok], pref * VS[ok] ** ex, "--", color=col, lw=0.9)
    sl = np.polyfit(np.log(VS[fit_rng]), np.log(m[fit_rng]), 1)[0]
    brng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        kk = brng.integers(0, NR, NR)
        mb = np.exp(np.nanmean(np.log(A[kk][:, fit_rng]), 0))
        bs.append(np.polyfit(np.log(VS[fit_rng]), np.log(mb), 1)[0])
    fits.append([hu, ex, sl, *np.percentile(bs, [2.5, 97.5])])
    write_csv(f"fig05a_area_H{hu}", {"volume": VS[ok], "area": m[ok]})
fits = np.array(fits)
write_csv("fig05a_fit", {"H": fits[:, 0], "expected": fits[:, 1],
                         "slope": fits[:, 2], "ci_lo": fits[:, 3],
                         "ci_hi": fits[:, 4]})
ax.set_xlabel("injected gas (L)")
ax.set_ylabel(r"area under gas (m$^2$)")
panel_label(ax, "(a)", "Single-source growth")

ax = axs[1]
tab = {"t_s": TS}
for tl, col in zip(TILTS, SEQ[1:]):
    S = cache[f"S{tl}"]
    q1, md, q3 = np.percentile(S, [25, 50, 75], axis=0)
    ax.fill_between(TS / 60, q1, q3, color=col, alpha=0.18, lw=0)
    ax.plot(TS / 60, md, color=col)
    tab[f"silt_g_median_tilt{tl:g}"] = md
    tab[f"silt_g_q1_tilt{tl:g}"] = q1
    tab[f"silt_g_q3_tilt{tl:g}"] = q3
write_csv("fig05b_hover_silt", tab)
ax.set_xlim(0, DUR / 60)
ax.set_ylim(bottom=0)
ax.set_xlabel("time holding station (min)")
ax.set_ylabel("silt released (g)")
panel_label(ax, "(b)", "Silt while hovering")

ax = axs[2]
ax2 = ax.twinx()
d = [cache[f"dist{tl}"] for tl in TILTS]
lo = [cache[f"lost{tl}"] for tl in TILTS]
ax.errorbar(TILTS, [np.median(v) for v in d],
            yerr=[[np.median(v) - np.percentile(v, 25) for v in d],
                  [np.percentile(v, 75) - np.median(v) for v in d]],
            fmt="o-", color="k", ms=3.5, mfc="none", capsize=2, lw=0.9)
ax2.plot(TILTS, [100 * np.median(v) for v in lo], "s--", color=C["gray"],
         ms=3.5, mfc="none", lw=0.9)
ax.set_xlabel(r"dip $\beta$ (deg)")
ax.set_ylabel("up-dip offset of silt (m)")
ax2.set_ylabel("gas leaving section (%)", color=C["gray"])
ax2.tick_params(axis="y", colors=C["gray"])
ax.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax.tick_params(right=False)
panel_label(ax, "(c)", "Offset and gas loss")
write_csv("fig05c_offset", {
    "tilt_deg": TILTS, "offset_median": [np.median(v) for v in d],
    "offset_q1": [np.percentile(v, 25) for v in d],
    "offset_q3": [np.percentile(v, 75) for v in d],
    "lost_median": [np.median(v) for v in lo]})

fig.tight_layout(w_pad=0.8)
hh, ll = handles([r"$H=0.5$", r"$H=0.8$", r"slope $2/(2+H)$"],
                 ["k", C["gray"], "k"], ["-", "-", "--"], ["o", "s", None])
legend_below(fig, axs[0], hh, ll, ncol=2)
hh, ll = handles([rf"$\beta={t:g}^\circ$" for t in TILTS], SEQ[1:5],
                 ["-"] * 4)
legend_below(fig, axs[1], hh, ll, ncol=4)
hh, ll = handles(["silt offset", "gas leaving"], ["k", C["gray"]],
                 ["-", "--"], ["o", "s"])
legend_below(fig, axs[2], hh, ll, ncol=2)
print(save(fig, "fig05_hover"))
print(fits)
for tl in TILTS:
    print(tl, np.median(cache[f"S{tl}"][:, -1]),
          np.median(cache[f"dist{tl}"]), np.median(cache[f"lost{tl}"]))
