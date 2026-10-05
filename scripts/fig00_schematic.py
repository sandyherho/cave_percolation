"""Figure 0. Model section and free-body diagrams.

(a) Section along the passage, to scale.  An idealized rough ceiling
h(x) dipping at beta holds gas pools with flat interfaces at levels L;
the up-dip pool is full to its spill point, and gas crossing the saddle
climbs along the ceiling to the next pool.  An open-water diver exhales
from the second stage at z_reg; silt stripped by the moving contact
lines settles at w_s and spreads with eddy diffusivity K; the eye band
is centered at z_eye; H_p is the ceiling-to-floor height.
(b) Free-body diagram of a gas pool: the water pressure p_w(L) acting on
the flat interface balances the ceiling reaction, which carries the gas
pressure p_g; at the rim the meniscus of radius r pins the contact line
until the gas thickness exceeds ell = 2 gamma/(rho g r).
(c) Free-body diagram of a loose grain of radius R on the ceiling as a
contact line passes: capillary force F_gamma, adhesion F_A, and net
weight W'.
(d) Free-body diagram of a settling grain of diameter d: net weight W'
balances the Stokes drag F_D at the settling velocity w_s.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon

from cave_percolation.plotting import setup, OKABE_ITO as C, panel_label
from cave_percolation.plotting import save
from cave_percolation.scenario import PRM
from cave_percolation.sketch import draw_diver

ROCK, ROCK_EDGE = "#cbbfa9", "#6d6352"
GASF, GASE = "#d9e8f0", "#6f95ab"
SILT = "#8c6a3f"
FORCE, VEL = "k", C["blue"]


def arrow(ax, x0, y0, x1, y1, color=FORCE, lw=1.1, ms=9, style="-|>",
          z=6):
    """Draw a force or velocity arrow in data coordinates."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def label(ax, x, y, s, **kw):
    """Place a math label with a white halo."""
    kw.setdefault("fontsize", 8.5)
    t = ax.text(x, y, s, zorder=8, **kw)
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6,
                    alpha=0.85))
    return t


def bare(ax):
    """Remove ticks and frame for a diagram panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


setup()
W_IN = 7.2
fig = plt.figure(figsize=(W_IN, 4.55))

# ------------------------------------------------------------ (a) section
L_A, Z_A = 9.0, 2.62
ax = fig.add_axes([0.01, 0.45, 0.98, 0.98 * W_IN * Z_A / L_A / 4.55])
BETA = np.radians(2.0)
x = np.linspace(0, L_A, 1800)
mean = 0.32 - x * np.tan(BETA)            # depth of the mean ceiling
bumps = (0.20 * np.exp(-((x - 4.0) / 0.55) ** 2)
         + 0.17 * np.exp(-((x - 6.9) / 0.45) ** 2)
         + 0.03 * np.sin(2 * np.pi * x / 0.9 + 0.7)
         + 0.02 * np.sin(2 * np.pi * x / 0.37))
d = mean - bumps                          # ceiling depth (down positive)
h = -d
top = d.min() - 0.12
floor = mean + PRM.h_passage
ax.fill_between(x, top, d, color=ROCK, lw=0, zorder=1)
ax.plot(x, d, color=ROCK_EDGE, lw=1.0, zorder=2)
ax.fill_between(x, floor, top + Z_A + 0.1, color=ROCK, lw=0, zorder=1)
ax.plot(x, floor, color=ROCK_EDGE, lw=1.0, zorder=2)


def pocket(xc, half):
    """Index window around the ceiling high near xc."""
    w = np.flatnonzero(np.abs(x - xc) < half)
    return w[np.argmax(h[w])], w


# full pool: level at its up-dip saddle (the higher of its two saddles)
i1, w1 = pocket(4.0, 1.4)
i2, w2 = pocket(6.9, 1.0)
sad = i1 + int(np.argmin(h[i1:i2]))
lev1 = h[sad]
m1 = (h >= lev1) & (np.arange(x.size) < sad) & (np.arange(x.size) > i1 - 300)
lev2 = h[i2] - 0.45 * (h[i2] - h[i2 + int(np.argmin(h[i2:i2 + 300]))])
m2 = (h >= lev2) & (np.abs(np.arange(x.size) - i2) < 200)
for m, lev in ((m1, lev1), (m2, lev2)):
    ax.fill_between(x, d, -lev, where=m, color=GASF, lw=0, zorder=3)
    xx = x[m]
    ax.plot([xx.min(), xx.max()], [-lev, -lev], color=GASE, lw=1.1, zorder=4)
ax.plot([x[sad]], [d[sad]], "o", ms=4, mfc="white", mec="k", zorder=7)
label(ax, x[sad] - 0.05, d[sad] - 0.14, "spill point", fontsize=7.5,
      ha="center")
ax.annotate("", xy=(x[i1], -lev1), xytext=(x[i1], d[i1]),
            arrowprops=dict(arrowstyle="<->", lw=0.7), zorder=6)
label(ax, x[i1] + 0.07, 0.5 * (d[i1] - lev1), r"$h-L$", fontsize=7.5,
      va="center")
label(ax, x[m2].max() + 0.08, -lev2, r"$L$", va="center")
# migrating gas: from the saddle along the ceiling to the next pool
j1 = int(np.flatnonzero(m2).min())
px = x[sad:j1]
pz = d[sad:j1] + 0.045
ax.plot(px, pz, color=VEL, lw=1.0, ls=(0, (3, 2)), zorder=5)
arrow(ax, px[-15], pz[-15], px[-1], pz[-1], color=VEL, ms=8)
label(ax, px[len(px) // 2], pz[len(px) // 2] + 0.16, "migrating gas",
      fontsize=7.5, color=VEL, ha="center")

# diver and exhaled bubbles, regulator at z_reg below the mean ceiling
xd = 2.7
zm = np.interp(xd, x, mean)
draw_diver(ax, xd, zm + PRM.z_reg, scale=1.0, kz=1.0, zorder=6)
zc = np.interp(xd + 0.02, x, d)
zb = np.linspace(zm + PRM.z_reg - 0.04, zc + 0.04, 9)
ax.scatter(xd + 0.02 + 0.02 * np.sin(np.arange(9)), zb,
           s=np.linspace(3, 15, 9), facecolor="none", edgecolor=VEL, lw=0.6,
           zorder=7)
ax.annotate("", xy=(xd + 0.38, zm + PRM.z_reg), xytext=(xd + 0.38, zm),
            arrowprops=dict(arrowstyle="<->", lw=0.7), zorder=6)
label(ax, xd + 0.45, zm + 0.5 * PRM.z_reg, r"$z_{\rm reg}$",
      va="center")

# eye band and mean ceiling, parallel to the dip
for off in (PRM.z_eye - 0.1, PRM.z_eye + 0.1):
    ax.plot(x, mean + off, color=C["gray"], lw=0.6, ls="--", zorder=2)
label(ax, 0.1, mean[0] + PRM.z_eye, "eye band", fontsize=7.5,
      color=C["gray"], va="center")

# falling silt under the stripped ceiling, settling and diffusion
rng = np.random.default_rng(3)
gx = rng.uniform(4.6, 6.4, 80)
gz = np.interp(gx, x, d) + 0.05 + 1.55 * rng.random(80) ** 1.5
ax.scatter(gx, gz, s=rng.choice([1.0, 3.0, 7.0], 80), color=SILT, lw=0,
           zorder=4)
arrow(ax, 6.75, 0.85, 6.75, 1.35, color=VEL, ms=8)
label(ax, 6.82, 1.1, r"$w_s$", color=VEL, va="center")
arrow(ax, 4.22, 1.55, 3.78, 1.55, color=C["gray"], ms=7, style="<|-|>")
arrow(ax, 4.0, 1.33, 4.0, 1.77, color=C["gray"], ms=7, style="<|-|>")
label(ax, 4.27, 1.47, r"$K$", color=C["gray"])

# passage height, dip, and axes
xh = 8.55
ax.annotate("", xy=(xh, np.interp(xh, x, mean)),
            xytext=(xh, np.interp(xh, x, floor)),
            arrowprops=dict(arrowstyle="<->", lw=0.7), zorder=6)
label(ax, xh - 0.08, np.interp(xh, x, mean) + 1.2, r"$H_p$", ha="right")
xb, lb = 5.2, 2.4
zb0 = np.interp(xb, x, floor) - 0.12
ax.plot([xb, xb + lb], [zb0, zb0], color="k", lw=0.6, ls=":")
ax.plot([xb, xb + lb], [zb0, zb0 - lb * np.tan(BETA)], color="k", lw=0.8)
ax.add_patch(Arc((xb, zb0), 2 * (lb - 0.15), 2 * (lb - 0.15),
                 theta1=0, theta2=np.degrees(BETA), lw=0.8))
label(ax, xb + lb + 0.05, zb0 - 0.5 * lb * np.tan(BETA), r"$\beta$",
      va="center")
label(ax, xb + 0.1, zb0 - 0.22, "floor and mean ceiling dip", fontsize=6.8,
      va="center", color=C["gray"])
label(ax, 1.2, np.interp(1.2, x, d) + 0.04, r"$h(x)$", va="top")
arrow(ax, 0.15, 2.0, 0.85, 2.0, ms=8)
label(ax, 0.9, 2.0, r"$x$ (up-dip)", va="center", fontsize=7.5)
arrow(ax, 0.15, 1.62, 0.15, 1.98, ms=8)
label(ax, 0.22, 1.78, r"$z$", va="center", fontsize=7.5)
ax.set_xlim(0, L_A)
ax.set_ylim(top + Z_A, top)
ax.set_aspect("equal")
bare(ax)
panel_label(ax, "(a)", "Model section along the passage, to scale",
            dy=1.01)

# common frame for the three free-body diagrams
FX, FZ = (-1.3, 1.3), (1.45, -0.35)
HB = (FZ[0] - FZ[1]) / (FX[1] - FX[0]) * 0.31 * W_IN / 4.55
POS = [[0.01, 0.02, 0.31, HB], [0.345, 0.02, 0.31, HB],
       [0.68, 0.02, 0.31, HB]]

# ------------------------------------------------------- (b) gas pool FBD
ax = fig.add_axes(POS[0])
u = np.linspace(-1.3, 1.3, 600)
ceil = 0.62 * (u / 1.2) ** 2 - 0.05
lev, ell = 0.42, 0.11
ax.fill_between(u, FZ[1], ceil, color=ROCK, lw=0)
ax.plot(u, ceil, color=ROCK_EDGE, lw=1.0)
m = ceil <= lev - ell
ue = np.array([u[m][0], u[m][-1]])
ax.fill_between(u, ceil, lev, where=m, color=GASF, lw=0)
ax.plot(ue, [lev, lev], color=GASE, lw=1.2)
for k, xe in enumerate(ue):
    sg = -1.0 if k == 0 else 1.0
    t = np.linspace(0, np.pi / 2, 20)
    ax.plot(xe + sg * ell * np.sin(t) * 0.6, lev - ell + ell * np.cos(t),
            color=GASE, lw=1.0)
for xx in np.linspace(ue[0] + 0.15, ue[1] - 0.15, 6):
    arrow(ax, xx, lev + 0.34, xx, lev + 0.03, ms=6, lw=0.8)
label(ax, 0.0, lev + 0.47, r"water pressure $p_w(L)=p_0-\rho gL$",
      ha="center", fontsize=7)
for xx in np.linspace(-0.45, 0.45, 4):
    yc = np.interp(xx, u, ceil)
    arrow(ax, xx, yc + 0.005, xx, yc + 0.17, ms=6, lw=0.8,
          color=C["vermil"])
label(ax, 0.0, 0.27, r"ceiling reaction $p_g$", ha="center", fontsize=7,
      color=C["vermil"])
xr = ue[1] + 0.13
ax.annotate("", xy=(xr, lev), xytext=(xr, lev - ell),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
label(ax, xr + 0.05, lev - 0.5 * ell, r"$\ell$", va="center", fontsize=8)
label(ax, -1.25, lev - 0.03, r"$L$", fontsize=8, va="center")
label(ax, 0.0, 1.27, r"$\ell=2\gamma/(\rho g r)$ pins the rim",
      ha="center", fontsize=7)
ax.set_xlim(*FX)
ax.set_ylim(*FZ)
ax.set_aspect("equal")
bare(ax)
panel_label(ax, "(b)", "Gas pool", dy=1.01)

# --------------------------------------------------- (c) contact-line FBD
ax = fig.add_axes(POS[1])
ax.fill_between(list(FX), FZ[1], 0.0, color=ROCK, lw=0)
ax.plot(list(FX), [0, 0], color=ROCK_EDGE, lw=1.0)
R = 0.27
ang = np.radians(235)
P = np.array([R * np.cos(ang), R - R * np.sin(ang) * -1.0])
P = np.array([R * np.cos(ang), R + R * -np.sin(ang)])
tq = np.linspace(0, 1, 60)
ix = P[0] + (FX[0] - P[0]) * tq
iz = P[1] + (0.95 - P[1]) * tq ** 1.3
arc_t = np.linspace(np.pi / 2, ang, 40)
arc = np.column_stack([R * np.cos(arc_t), R - R * np.sin(arc_t)])
gas_poly = np.vstack([[[FX[0], 0.0], [0.0, 0.0]], arc,
                      np.column_stack([ix, iz]), [[FX[0], 0.95]]])
ax.add_patch(Polygon(gas_poly, closed=True, facecolor=GASF,
                     edgecolor="none", zorder=2))
ax.plot(ix, iz, color=GASE, lw=1.2, zorder=3)
ax.add_patch(Circle((0.0, R), R, facecolor="#b08a58", edgecolor="#6b5232",
                    lw=0.8, zorder=4))
tang = np.array([ix[8] - ix[0], iz[8] - iz[0]])
tang /= np.hypot(*tang)
arrow(ax, P[0], P[1], P[0] + 0.55 * tang[0], P[1] + 0.55 * tang[1], ms=8)
label(ax, P[0] + 0.55 * tang[0] - 0.05, P[1] + 0.55 * tang[1] + 0.13,
      r"$F_\gamma\approx2\pi R\gamma$", fontsize=7.5, ha="center")
arrow(ax, 0.0, R - 0.02, 0.0, 0.03, ms=7, color=C["green"])
label(ax, 0.08, 0.13, r"$F_A$", color=C["green"], fontsize=8)
arrow(ax, 0.12, 2 * R, 0.12, 2 * R + 0.33, ms=8)
label(ax, 0.19, 2 * R + 0.25, r"$W'$", fontsize=8)
arrow(ax, -1.15, 0.22, -0.6, 0.22, ms=7, color=VEL)
label(ax, -1.2, 0.38, "advancing gas", fontsize=6.8, color=VEL)
label(ax, 0.7, 0.8, "water", fontsize=7, color=C["gray"])
ax.annotate("", xy=(R + 0.08, 0.0), xytext=(R + 0.08, R),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
label(ax, R + 0.13, 0.15, r"$R$", fontsize=8, va="center")
label(ax, 0.0, 1.27, r"detaches when $F_\gamma>F_A+W'$", ha="center",
      fontsize=7)
ax.set_xlim(*FX)
ax.set_ylim(*FZ)
ax.set_aspect("equal")
bare(ax)
panel_label(ax, "(c)", "Grain under a contact line", dy=1.01)

# ------------------------------------------------------ (d) settling FBD
ax = fig.add_axes(POS[2])
zc0 = 0.45
ax.add_patch(Circle((0, zc0), 0.2, facecolor="#b08a58",
                    edgecolor="#6b5232", lw=0.8, zorder=4))
arrow(ax, 0.0, zc0 + 0.21, 0.0, zc0 + 0.6, ms=9)
label(ax, 0.08, zc0 + 0.5, r"$W'=(\rho_s-\rho)\,g\,\pi d^3/6$",
      fontsize=7, va="center")
arrow(ax, 0.0, zc0 - 0.21, 0.0, zc0 - 0.6, ms=9, color=C["vermil"])
label(ax, 0.08, zc0 - 0.5, r"$F_D=3\pi\mu d\,w_s$", fontsize=7,
      color=C["vermil"], va="center")
arrow(ax, -0.75, zc0 - 0.25, -0.75, zc0 + 0.25, ms=8, color=VEL)
label(ax, -1.05, zc0, r"$w_s$", color=VEL, va="center")
ax.annotate("", xy=(-0.2, zc0), xytext=(0.2, zc0),
            arrowprops=dict(arrowstyle="<->", lw=0.6), zorder=5)
label(ax, 0.27, zc0, r"$d$", fontsize=8, va="center")
label(ax, 0.0, 1.27, r"$W'=F_D\ \Rightarrow\ w_s=(\rho_s-\rho)gd^2/(18\mu)$",
      ha="center", fontsize=7)
ax.set_xlim(*FX)
ax.set_ylim(*FZ)
ax.set_aspect("equal")
bare(ax)
panel_label(ax, "(d)", "Settling grain", dy=1.01)

print(save(fig, "fig00_schematic"))
