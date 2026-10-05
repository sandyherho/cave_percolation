"""Side-view silhouette of an open-water scuba diver for schematics.

The diver is drawn in profile in horizontal trim, facing +x, with a
single aluminum 80 cylinder (0.66 m long, 0.18 m diameter) on a jacket
buoyancy compensator, valve and first stage, a low-pressure hose to the
second stage at the mouth, mask, hood, weight belt, and open-heel fins.
Shapes are defined in a local frame in meters (u forward, v up) and
mapped to data coordinates with an aspect correction, so the figure is
not distorted on axes with unequal scales.  The mouthpiece, where the
exhaled gas leaves, is placed exactly at the given point.
"""

import numpy as np
from matplotlib.patches import Polygon

__all__ = ["draw_diver", "aspect_ratio", "MOUTH"]

MOUTH = (0.885, -0.055)

LIGHT = dict(suit="#2b3440", far="#1c232c", bcd="#46566b", tank="#9aa6b2",
             tank_edge="#5d6875", metal="#c9ced4", glass="#a9cbe0",
             fin="#141a21", belt="#59606a", hose="#11161c")
DARK = dict(suit="#8fa3b8", far="#5e6f82", bcd="#6f8297", tank="#c3ccd6",
            tank_edge="#8b97a4", metal="#e3e7eb", glass="#d6ecf7",
            fin="#4b5a6a", belt="#7d8794", hose="#a6b4c2")


def aspect_ratio(ax):
    """Return data units of y per data unit of x that look equal on screen."""
    fig = ax.figure
    bb = ax.get_position()
    w = bb.width * fig.get_figwidth()
    h = bb.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (abs(y1 - y0) / h) / (abs(x1 - x0) / w)


def _ellipse(cu, cv, ru, rv, n=48, a0=0.0, a1=2 * np.pi):
    t = np.linspace(a0, a1, n)
    return np.column_stack([cu + ru * np.cos(t), cv + rv * np.sin(t)])


def _stroke(pts, widths):
    """Polygon of a tapered stroke along a polyline."""
    p = np.asarray(pts, float)
    w = np.interp(np.linspace(0, 1, len(p)), np.linspace(0, 1, len(widths)),
                  widths)
    d = np.gradient(p, axis=0)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    left = p + 0.5 * w[:, None] * nrm
    right = p - 0.5 * w[:, None] * nrm
    return np.vstack([left, right[::-1]])


def _rounded_rect(u0, u1, vc, r, n=16):
    """Cylinder side view: rectangle with hemispherical ends."""
    a = _ellipse(u1 - r, vc, r, r, n, -np.pi / 2, np.pi / 2)
    b = _ellipse(u0 + r, vc, r, r, n, np.pi / 2, 3 * np.pi / 2)
    return np.vstack([a, b])


def _bezier(p0, p1, p2, n=30):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) \
        + t ** 2 * np.asarray(p2)


def _parts():
    """List of (polygon in local meters, color key, zorder offset)."""
    out = []
    for dv, du, key in ((0.045, -0.07, "far"), (0.0, 0.0, "suit")):
        hip, knee, ankle = (-0.08, -0.01), (-0.55, -0.045), (-0.92, 0.09)
        leg = [hip, (-0.33, -0.03), knee, (-0.74, 0.02), ankle]
        leg = [(u + du, v + dv) for u, v in leg]
        out.append((_stroke(leg, [0.20, 0.155, 0.115, 0.09, 0.07]), key,
                    0))
        fin = np.array([(-0.88, 0.125), (-0.97, 0.15), (-1.30, 0.235),
                        (-1.47, 0.27), (-1.50, 0.215), (-1.33, 0.17),
                        (-0.97, 0.055), (-0.88, 0.06)])
        fin = fin + np.array([du, dv])
        out.append((fin, "fin", 0.1))
    torso = np.vstack([_ellipse(0.20, 0.0, 0.40, 0.145, 60)])
    out.append((torso, "suit", 0.2))
    bcd = np.array([(-0.08, 0.06), (0.02, 0.15), (0.50, 0.17), (0.60, 0.10),
                    (0.58, -0.02), (0.48, -0.08), (0.40, -0.02),
                    (0.30, 0.05), (0.02, 0.04)])
    out.append((bcd, "bcd", 0.3))
    out.append((_rounded_rect(-0.11, 0.55, 0.255, 0.092), "tank", 0.4))
    out.append((np.array([(0.55, 0.235), (0.62, 0.235), (0.62, 0.275),
                          (0.55, 0.275)]), "metal", 0.45))
    out.append((_ellipse(0.655, 0.255, 0.04, 0.04, 24), "metal", 0.45))
    out.append((np.array([(-0.06, 0.035), (0.02, 0.035), (0.02, -0.13),
                          (-0.06, -0.13)]), "belt", 0.5))
    arm_far = [(0.47, -0.02), (0.36, -0.16), (0.56, -0.19)]
    out.append((_stroke(arm_far, [0.075, 0.065, 0.05]), "far", 0.15))
    neck = np.array([(0.56, 0.06), (0.68, 0.09), (0.70, -0.04),
                     (0.57, -0.07)])
    out.append((neck, "suit", 0.5))
    out.append((_ellipse(0.745, 0.035, 0.105, 0.10, 40), "suit", 0.55))
    mask = np.array([(0.80, 0.075), (0.86, 0.095), (0.885, 0.06),
                     (0.875, -0.005), (0.82, -0.015)])
    out.append((mask, "glass", 0.6))
    hose = _bezier((0.66, 0.24), (0.95, 0.30), (0.88, -0.02))
    out.append((_stroke(hose, [0.018, 0.018]), "hose", 0.65))
    out.append((_ellipse(MOUTH[0], MOUTH[1], 0.04, 0.035, 24), "hose",
                0.7))
    arm = [(0.47, -0.04), (0.33, -0.19), (0.55, -0.215)]
    out.append((_stroke(arm, [0.085, 0.07, 0.055]), "suit", 0.75))
    return out


def draw_diver(ax, mouth_x, mouth_z, scale=1.0, kz=1.0, palette="light",
               zorder=5, depth_down=True):
    """Draw the diver with its mouthpiece at (mouth_x, mouth_z).

    ``kz`` is the result of :func:`aspect_ratio`; ``depth_down`` is True
    when the y axis is depth increasing downward.  Returns the list of
    patches.
    """
    col = LIGHT if palette == "light" else DARK
    sgn = -1.0 if depth_down else 1.0
    pats = []
    for poly, key, dz in _parts():
        u = poly[:, 0] - MOUTH[0]
        v = poly[:, 1] - MOUTH[1]
        xy = np.column_stack([mouth_x + scale * u,
                              mouth_z + sgn * scale * kz * v])
        edge = col["tank_edge"] if key == "tank" else "none"
        p = Polygon(xy, closed=True, facecolor=col[key], edgecolor=edge,
                    lw=0.4, zorder=zorder + dz)
        ax.add_patch(p)
        pats.append(p)
    return pats
