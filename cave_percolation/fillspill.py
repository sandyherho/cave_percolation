"""Quasi-static fill-spill-merge of buoyant gas under a ceiling.

Derivation.  Water pressure is hydrostatic, p_w(z) = p0 - rho g z, and
the gas density is negligible, so a connected gas pool has uniform
pressure p_g and a flat gas-water interface at the level L where
p_g = p_w(L).  A cell of effective ceiling elevation h_eff is covered
when L <= h_eff, and the pool volume is

    V = a^2 sum_{i in pool} (h_eff_i - L).

Injecting gas lowers L.  The pool grows by the perimeter cell with the
highest h_eff, which is the invasion-percolation rule with threshold
-h_eff.  When that cell j has a neighbor outside the pool that lies
higher than j, the pool is full: further gas crosses the saddle j and
rises along the steepest-ascent path of h_eff to a crest, where it
starts or joins another pool.  Two pools whose levels meet at a common
saddle become one.  Gas that reaches an open end of the passage leaves
the domain.  This is the fill-spill-merge construction used for
depression hierarchies, applied to the inverted ceiling.

The engine is event driven and exact: volume is conserved to rounding,
and when every pool is full the covered set equals the priority-flood
filling of -h_eff (checked in the verification figure).

Silt bookkeeping.  A contact line crossing a cell for the first time
removes a fraction phi_c of its loose silt.  Gas volume v traveling
along a migration path strips a fraction 1 - exp(-v/v_d) from each cell
it passes, which is independent of how the gas is split into parcels.
"""

import heapq
import math

import numpy as np

__all__ = ["FillSpill", "priority_flood", "pool_stats"]


class _Pool:
    __slots__ = ("id", "n", "S", "L", "heap", "full_at", "cells")

    def __init__(self, pid, cell, h):
        self.id = pid
        self.n = 1
        self.S = h
        self.L = h
        self.heap = []
        self.full_at = None
        self.cells = [cell]


def _lattice(ny, nx):
    """4-neighbor table on an (ny, nx) lattice with exit nodes at x ends.

    Exit node ``N + iy`` sits beyond the down-dip end of row iy and node
    ``N + ny + iy`` beyond the up-dip end.  The y boundaries are closed.
    """
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    nb = -np.ones((n, 4), dtype=np.int64)
    nb[idx[:, 1:].ravel(), 0] = idx[:, :-1].ravel()
    nb[idx[:, 0].ravel(), 0] = n + np.arange(ny)
    nb[idx[:, :-1].ravel(), 1] = idx[:, 1:].ravel()
    nb[idx[:, -1].ravel(), 1] = n + ny + np.arange(ny)
    nb[idx[1:, :].ravel(), 2] = idx[:-1, :].ravel()
    nb[idx[:-1, :].ravel(), 3] = idx[1:, :].ravel()
    return nb


class FillSpill:
    """Fill-spill-merge engine on an effective ceiling ``h_eff`` (ny, nx).

    Parameters
    ----------
    h_eff : array (ny, nx)
        Effective ceiling elevation (m), +x up-dip.
    a : float
        Lattice spacing (m).
    tanb : float
        Dip used to place the exit nodes one cell beyond each open end.
    silt : array (ny, nx) or None
        Loose silt mass per cell (g); modified in place copy.
    phi_c, v_d : float
        Contact-line and path stripping constants.
    """

    def __init__(self, h_eff, a, tanb=0.0, silt=None, phi_c=0.5,
                 v_d=2e-6):
        """Build the lattice, exit nodes, and steepest-ascent table."""
        ny, nx = h_eff.shape
        self.ny, self.nx, self.a = ny, nx, a
        self.a2 = a * a
        n = nx * ny
        self.N = n
        he = np.concatenate([h_eff.ravel(),
                             h_eff[:, 0] - a * tanb,
                             h_eff[:, -1] + a * tanb])
        nb = _lattice(ny, nx)
        valid = nb >= 0
        hn = np.where(valid, he[np.where(valid, nb, 0)], -np.inf)
        best = np.argmax(hn, axis=1)
        top = nb[np.arange(n), best]
        up = np.where(hn[np.arange(n), best] > he[:n], top, -1)
        self.he = he.tolist()
        self.nb = [[c for c in row if c >= 0] for row in nb.tolist()]
        self.up = up.tolist()
        self.pid = [-1] * (n + 2 * ny)
        self.pools = {}
        self._next = 0
        self.injected = 0.0
        self.exported = 0.0
        self.silt = None if silt is None else silt.ravel().astype(float)
        self.phi_c, self.v_d = phi_c, v_d
        self.flux = np.zeros(n)
        self.t_cover = np.full(n, np.nan)
        self.rel_t, self.rel_c, self.rel_m, self.rel_tag = [], [], [], []
        self.n_merge = 0
        self.n_spill = 0
        self._t = 0.0
        self._tag = 0

    # ---------------------------------------------------------- bookkeeping
    def _release(self, c, m):
        if m > 0.0:
            self.rel_t.append(self._t)
            self.rel_c.append(c)
            self.rel_m.append(m)
            self.rel_tag.append(self._tag)

    def _contact(self, c):
        self.t_cover[c] = self._t
        if self.silt is not None:
            m = self.phi_c * self.silt[c]
            self.silt[c] -= m
            self._release(c, m)

    def _sweep(self, path, dv):
        if not path:
            return
        self.flux[path] += dv
        if self.silt is not None:
            f = 1.0 - math.exp(-dv / self.v_d)
            for c in path:
                m = f * self.silt[c]
                self.silt[c] -= m
                self._release(c, m)

    # --------------------------------------------------------------- pools
    def _new_pool(self, c):
        p = _Pool(self._next, c, self.he[c])
        self._next += 1
        self.pools[p.id] = p
        self.pid[c] = p.id
        self._push_nbrs(p, c)
        self._contact(c)
        return p

    def _push_nbrs(self, p, c):
        he, pid = self.he, self.pid
        for m in self.nb[c]:
            if pid[m] != p.id:
                heapq.heappush(p.heap, (-he[m], m))

    def _peek(self, p):
        pid, heap = self.pid, p.heap
        while heap:
            c = heap[0][1]
            if c < self.N and pid[c] == p.id:
                heapq.heappop(heap)
                continue
            if c < self.N and pid[c] >= 0:
                raise RuntimeError("pools touch without a merge")
            return c
        raise RuntimeError("pool perimeter exhausted")

    def _is_spill(self, p, c):
        he, pid = self.he, self.pid
        hc = he[c]
        for m in self.nb[c]:
            if pid[m] != p.id and he[m] > hc:
                return True
        return False

    def _route(self, p, j):
        """Steepest-ascent path from saddle j away from pool p."""
        he, pid, up = self.he, self.pid, self.up
        best, hb = -1, he[j]
        for m in self.nb[j]:
            if pid[m] != p.id and he[m] > hb:
                best, hb = m, he[m]
        path = [j]
        c = best
        while True:
            if c >= self.N or pid[c] >= 0:
                return path, c
            u = up[c]
            if u < 0:
                return path, c
            path.append(c)
            c = u

    def _ascend(self, c):
        """Steepest-ascent path from an uncovered cell."""
        pid, up = self.pid, self.up
        path = []
        while True:
            if c >= self.N or pid[c] >= 0:
                return path, c
            u = up[c]
            if u < 0:
                return path, c
            path.append(c)
            c = u

    def _merge(self, p, q, j):
        big, small = (p, q) if p.n >= q.n else (q, p)
        pid = self.pid
        for c in small.cells:
            pid[c] = big.id
        big.cells.extend(small.cells)
        big.n += small.n
        big.S += small.S
        big.L = self.he[j]
        for e in small.heap:
            heapq.heappush(big.heap, e)
        big.full_at = None
        del self.pools[small.id]
        self.n_merge += 1
        return big

    def _deliver(self, end, dv):
        """Deliver gas that arrives at the end of a migration path."""
        if end >= self.N:
            self.exported += dv
            return
        q = self.pid[end]
        p = self.pools[q] if q >= 0 else self._new_pool(end)
        self._fill(p, dv)

    def _fill(self, p, dv):
        he, a2, pid = self.he, self.a2, self.pid
        while dv > 0.0:
            if p.full_at is not None:
                j = p.full_at
                if j >= self.N:
                    self.exported += dv
                    return
                path, end = self._route(p, j)
                self._sweep(path, dv)
                self.n_spill += 1
                if end >= self.N:
                    self.exported += dv
                    return
                q = pid[end]
                if q < 0:
                    p = self._new_pool(end)
                    continue
                qq = self.pools[q]
                if qq.L <= he[j] + 1e-12:
                    p = self._merge(p, qq, j)
                    continue
                p = qq
                continue
            c = self._peek(p)
            cap = (p.L - he[c]) * p.n * a2
            if dv <= cap:
                p.L -= dv / (p.n * a2)
                return
            dv -= cap
            p.L = he[c]
            if c >= self.N or self._is_spill(p, c):
                p.full_at = c
                continue
            heapq.heappop(p.heap)
            pid[c] = p.id
            p.n += 1
            p.S += he[c]
            p.cells.append(c)
            self._push_nbrs(p, c)
            self._contact(c)

    # ---------------------------------------------------------------- API
    def inject(self, cell, dv, t=0.0, tag=0):
        """Release gas volume ``dv`` (m^3) under lattice cell ``cell``."""
        self._t, self._tag = t, tag
        self.injected += dv
        if self.pid[cell] >= 0:
            self._fill(self.pools[self.pid[cell]], dv)
            return
        path, end = self._ascend(cell)
        self._sweep(path, dv)
        self._deliver(end, dv)

    def stored(self):
        """Total gas volume held in pools (m^3)."""
        return sum((p.S - p.n * p.L) * self.a2 for p in self.pools.values())

    def thickness(self):
        """Gas thickness h_eff - L under each cell (m), 0 where uncovered."""
        pid = np.asarray(self.pid[:self.N])
        lev = np.zeros(max(self._next, 1))
        for p in self.pools.values():
            lev[p.id] = p.L
        he = np.asarray(self.he[:self.N])
        t = np.where(pid >= 0, he - lev[np.maximum(pid, 0)], 0.0)
        return t.reshape(self.ny, self.nx)

    def covered(self):
        """Boolean map of covered cells."""
        return (np.asarray(self.pid[:self.N]) >= 0).reshape(self.ny, self.nx)

    def releases(self):
        """Arrays (t, cell, mass, tag) of every silt release."""
        return (np.asarray(self.rel_t), np.asarray(self.rel_c, dtype=int),
                np.asarray(self.rel_m), np.asarray(self.rel_tag, dtype=int))

    def all_full(self):
        """Return True when every pool is at its spill level."""
        return all(p.full_at is not None for p in self.pools.values())


def priority_flood(h_eff, a, tanb=0.0):
    """Saturated gas thickness by priority-flood filling of -h_eff.

    An independent algorithm for the state in which every pool is full:
    outlets are the exit nodes at the two open ends, and the filled
    surface of f = -h_eff gives the gas thickness F - f.
    """
    ny, nx = h_eff.shape
    n = nx * ny
    f = np.concatenate([-h_eff.ravel(), -(h_eff[:, 0] - a * tanb),
                        -(h_eff[:, -1] + a * tanb)]).tolist()
    nb = _lattice(ny, nx).tolist()
    filled = list(f)
    closed = [False] * (n + 2 * ny)
    heap = []
    for e in range(n, n + 2 * ny):
        closed[e] = True
        heapq.heappush(heap, (f[e], e))
    exits = {}
    for c in range(n):
        for m in nb[c]:
            if m >= n:
                exits.setdefault(m, []).append(c)
    while heap:
        fc, c = heapq.heappop(heap)
        nbrs = exits.get(c, []) if c >= n else [m for m in nb[c] if m >= 0]
        for m in nbrs:
            if not closed[m]:
                closed[m] = True
                filled[m] = max(f[m], fc)
                heapq.heappush(heap, (filled[m], m))
    fa = np.asarray(filled[:n])
    return (fa - np.asarray(f[:n])).reshape(ny, nx)


def pool_stats(thickness, a):
    """Areas (m^2) and volumes (m^3) of the connected gas pools."""
    from scipy import ndimage
    lab, n = ndimage.label(thickness > 0)
    if n == 0:
        return np.empty(0), np.empty(0)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(thickness), lab, idx) * a * a
    vol = ndimage.sum(thickness, lab, idx) * a * a
    return np.asarray(area), np.asarray(vol)
