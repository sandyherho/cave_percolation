"""Classic invasion percolation, the connected capillary limit.

On a smooth, nearly flat ceiling the relief is negligible against the
pinning heights and the gas stays connected.  Its pressure is uniform, so
the advancing contact line always moves into the perimeter cell with the
smallest entry threshold: invasion percolation without trapping (water
enclosed by gas on a ceiling stays connected to the water below, so no
cluster of uninvaded cells is ever cut off).  In this limit the invaded
set is a critical percolation cluster with fractal dimension 91/48, and
the thresholds it accepts are bounded by the site percolation threshold
p_c = 0.592746 of the square lattice.

A uniform dip adds a destabilizing gradient, thresholds u - G y with u
uniform on (0, 1).  The invasion then forms a single finger whose width
scales as G^(-nu/(1 + nu)) = G^(-4/7), with nu = 4/3.
"""

import heapq

import numpy as np

__all__ = ["invade", "mass_radius", "row_width", "P_C", "D_F", "NU"]

P_C = 0.59274621          # site percolation, square lattice
D_F = 91.0 / 48.0          # fractal dimension of the incipient cluster
NU = 4.0 / 3.0             # correlation-length exponent in two dimensions


def invade(thr, seeds, stop_edge=True, max_steps=None, stop_row=None):
    """Invade a 2D threshold field from ``seeds`` (flat indices).

    Returns the invasion order (flat indices) and the accepted thresholds.
    The run stops when a site on the lattice edge is invaded (point-source
    runs), when row ``stop_row`` is reached (gradient runs), or after
    ``max_steps``.
    """
    ny, nx = thr.shape
    t = thr.ravel().tolist()
    seen = bytearray(nx * ny)
    heap = []
    for s in seeds:
        seen[s] = 1
        heapq.heappush(heap, (t[s], s))
    order, acc = [], []
    max_steps = max_steps or nx * ny
    while heap and len(order) < max_steps:
        v, c = heapq.heappop(heap)
        order.append(c)
        acc.append(v)
        iy, ix = divmod(c, nx)
        if stop_edge and (ix == 0 or iy == 0 or ix == nx - 1
                          or iy == ny - 1):
            break
        if stop_row is not None and iy >= stop_row:
            break
        for m, ok in ((c - 1, ix > 0), (c + 1, ix < nx - 1),
                      (c - nx, iy > 0), (c + nx, iy < ny - 1)):
            if ok and not seen[m]:
                seen[m] = 1
                heapq.heappush(heap, (t[m], m))
    return np.asarray(order), np.asarray(acc)


def mass_radius(order, nx, center, radii):
    """Count invaded sites within each radius of ``center``."""
    iy, ix = np.divmod(order, nx)
    r = np.hypot(ix - center[1], iy - center[0])
    r.sort()
    return np.searchsorted(r, radii, side="right")


def row_width(order, nx, rows):
    """Return the rms lateral width of the invaded set in each row."""
    iy, ix = np.divmod(order, nx)
    w = np.full(len(rows), np.nan)
    for k, y in enumerate(rows):
        xs = ix[iy == y]
        if xs.size > 1:
            w[k] = xs.std()
    return w
