"""Figure 6. Who strips the ceiling: position in the team.

Reference passage (12.8 m x 3.2 m, dip 2 deg, arch 0.15 m, sigma1 = 0.10
m, H = 0.8, pinning on), twelve ceilings.  Divers swim up-dip along the
centerline at 0.2 m/s, 20 s apart.  A closed-circuit rebreather exhales no
gas in level swimming and releases no silt in this model.
(a) Silt released by each diver of a team of four, attributed to the
diver whose gas swept the ceiling, for RMV 20 and 40 L/min; median and
interquartile range.
(b) Total silt released by teams of one to four divers; dashed lines are
N times the median of a solo diver.
(c) Along-passage profile of released silt, g per meter of passage, for
a team of three (RMV 20) and one diver holding station for 5 min at
x = 6.4 m; medians.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from cave_percolation.experiments import team_run, hover_run
from cave_percolation.io_utils import write_csv, save_cache, load_cache
from cave_percolation.plotting import (setup, OKABE_ITO as C, panel_label,
                                       legend_below, handles, save)
from cave_percolation.scenario import PRM

RMVS = [20.0, 40.0]
NS = [1, 2, 3, 4]
XB = np.linspace(0, PRM.length, 65)

cache = load_cache("fig06")
if cache is None:
    out = {}
    for rmv in RMVS:
        per = np.zeros((len(PRM.seeds), 4))
        tot = np.zeros((len(PRM.seeds), len(NS)))
        for k, s in enumerate(PRM.seeds):
            for j, n in enumerate(NS):
                _, _, eng = team_run(PRM, s, n_divers=n, rmv=rmv)
                t, c, m, tag = eng.releases()
                tot[k, j] = m.sum()
                if n == 4:
                    per[k] = [m[tag == d].sum() for d in range(4)]
                if n == 3 and rmv == 20.0:
                    x = (c % PRM.nx + 0.5) * PRM.a
                    out.setdefault("prof_team", []).append(
                        np.histogram(x, XB, weights=m)[0] / np.diff(XB))
        out[f"per{rmv:g}"] = per
        out[f"tot{rmv:g}"] = tot
    for s in PRM.seeds:
        _, _, eng = hover_run(PRM, s, 6.4, 300.0)
        t, c, m, _ = eng.releases()
        x = (c % PRM.nx + 0.5) * PRM.a
        out.setdefault("prof_hover", []).append(
            np.histogram(x, XB, weights=m)[0] / np.diff(XB))
    out = {k: np.asarray(v) for k, v in out.items()}
    save_cache("fig06", **out)
    cache = load_cache("fig06")

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6),
                        gridspec_kw={"width_ratios": [1, 1, 1.35]})
cols = {20.0: "k", 40.0: C["vermil"]}
off = {20.0: -0.1, 40.0: 0.1}
tab_a, tab_b = {"position": np.arange(1, 5)}, {"n_divers": NS}
for rmv in RMVS:
    per = cache[f"per{rmv:g}"]
    q1, md, q3 = np.percentile(per, [25, 50, 75], axis=0)
    axs[0].errorbar(np.arange(1, 5) + off[rmv], md, yerr=[md - q1, q3 - md],
                    fmt="o", color=cols[rmv], ms=4, mfc="none", capsize=2,
                    lw=0.9)
    tab_a.update({f"median_rmv{rmv:g}": md, f"q1_rmv{rmv:g}": q1,
                  f"q3_rmv{rmv:g}": q3})
    tot = cache[f"tot{rmv:g}"]
    q1, md, q3 = np.percentile(tot, [25, 50, 75], axis=0)
    axs[1].errorbar(NS, md, yerr=[md - q1, q3 - md], fmt="o-",
                    color=cols[rmv], ms=4, mfc="none", capsize=2, lw=0.9)
    axs[1].plot(NS, md[0] * np.array(NS), "--", color=cols[rmv], lw=0.8)
    tab_b.update({f"median_rmv{rmv:g}": md, f"q1_rmv{rmv:g}": q1,
                  f"q3_rmv{rmv:g}": q3})
write_csv("fig06a_position", tab_a)
write_csv("fig06b_team_size", tab_b)
axs[0].set_xticks([1, 2, 3, 4])
axs[0].set_xlabel("position in team")
axs[0].set_ylabel("silt released (g)")
axs[0].set_ylim(bottom=0)
panel_label(axs[0], "(a)", "By team position")
axs[1].set_xticks(NS)
axs[1].set_xlabel("divers in team")
axs[1].set_ylabel("total silt released (g)")
axs[1].set_ylim(bottom=0)
panel_label(axs[1], "(b)", "By team size")

xc = 0.5 * (XB[1:] + XB[:-1])
pt = np.median(cache["prof_team"], 0)
ph = np.median(cache["prof_hover"], 0)
axs[2].step(xc, pt, where="mid", color="k", lw=1.0)
axs[2].step(xc, ph, where="mid", color=C["blue"], lw=1.0)
axs[2].axvline(6.4, color=C["blue"], ls=":", lw=0.8)
axs[2].set_xlim(0, PRM.length)
axs[2].set_ylim(bottom=0)
axs[2].set_xlabel(r"$x$ along passage (m), up-dip $\rightarrow$")
axs[2].set_ylabel(r"silt released (g m$^{-1}$)")
panel_label(axs[2], "(c)", "Along the passage")
write_csv("fig06c_profile", {"x": xc, "team3_g_per_m": pt,
                             "hover5min_g_per_m": ph})

fig.tight_layout(w_pad=0.8)
hh, ll = handles(["RMV 20 L/min", "RMV 40 L/min", r"$N\times$ solo"],
                 ["k", C["vermil"], C["gray"]], ["-", "-", "--"],
                 ["o", "o", None])
legend_below(fig, axs[:2], hh, ll, ncol=3)
hh, ll = handles(["team of 3", "hover 5 min"], ["k", C["blue"]],
                 ["-", "-"])
legend_below(fig, axs[2], hh, ll, ncol=2)
print(save(fig, "fig06_team"))
for rmv in RMVS:
    print(rmv, np.median(cache[f"per{rmv:g}"], 0),
          np.median(cache[f"tot{rmv:g}"], 0))
print("team3 total", np.median(cache["prof_team"].sum(1) * np.diff(XB)[0]),
      "hover", np.median(cache["prof_hover"].sum(1) * np.diff(XB)[0]))
