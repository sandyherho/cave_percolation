"""Silt settling in the passage and the diver's sighting range.

Transport.  Detached grains of diameter d settle at the Stokes velocity
w = (rho_s - rho) g d^2 / (18 mu) and spread by an eddy diffusivity K in a
width-averaged (x, z) section of the passage, z measured down from the
ceiling.  The concentration obeys

    dC/dt + U dC/dx + w dC/dz = K (d2C/dx2 + d2C/dz2),

with no flux through the ceiling and deposition at the floor z = H_p.
It is solved by Lagrangian particles with the exact Gaussian increment
of the free-space problem per step, reflection at the ceiling, and
absorption at the floor.  The mean residence time of a grain released at
the ceiling is, exactly,

    T = H_p/w - (K/w^2) (1 - exp(-w H_p / K)),

which the verification figure uses to test the random walk.

Optics.  For grains much larger than the wavelength the extinction
efficiency tends to 2, so the mass-specific beam attenuation is
b_m = 3/(rho_s d).  The beam attenuation is c = c_w + sum_k b_k C_k, and
the black-target sighting range is 4.8/c (Zaneveld and Pegau, 2003).
"""

import numpy as np

__all__ = ["stokes", "mass_attenuation", "size_fractions", "mfpt",
           "Column", "fallout", "beam_attenuation",
           "sighting_along", "sighting_range"]


def stokes(prm, d):
    """Stokes settling velocity (m s^-1) for diameter ``d`` (m)."""
    return (prm.rho_s - prm.rho) * prm.g * np.asarray(d) ** 2 / (18 * prm.mu)


def mass_attenuation(prm, d):
    """Mass-specific beam attenuation 3/(rho_s d) in m^2 g^-1."""
    return 3.0 / (prm.rho_s * 1e3 * np.asarray(d))


def size_fractions(prm, d_um):
    """Mass fractions of log-spaced classes from a log-normal distribution.

    Each class carries the mass between the geometric midpoints of its
    neighbors; the tails beyond the end classes are folded into them.
    """
    from math import erf, log, sqrt
    ld = np.log(np.asarray(d_um, float))
    edges = np.concatenate([[-np.inf], 0.5 * (ld[1:] + ld[:-1]), [np.inf]])
    mu = log(prm.d_median_um)

    def cdf(v):
        if np.isinf(v):
            return 0.0 if v < 0 else 1.0
        return 0.5 * (1 + erf((v - mu) / (prm.d_lnsd * sqrt(2))))
    return np.array([cdf(edges[k + 1]) - cdf(edges[k])
                     for k in range(len(ld))])


def mfpt(h, w, kappa):
    """Mean time for a grain released at a reflecting ceiling to deposit."""
    pe = w * h / kappa
    return h / w - kappa / w ** 2 * (-np.expm1(-pe))


class Column:
    """Lagrangian particles in a width-averaged passage section.

    Particles carry mass (g) of one size class; the concentration is
    mass / (W dx dz) with W the passage width.
    """

    def __init__(self, prm, w, rng, kappa=None, u=None):
        """Store settling velocities and start with no particles."""
        self.prm = prm
        self.w = np.asarray(w, float)
        self.kappa = prm.kappa if kappa is None else kappa
        self.u = prm.u_flow if u is None else u
        self.rng = rng
        self.x = np.empty(0)
        self.z = np.empty(0)
        self.m = np.empty(0)
        self.k = np.empty(0, dtype=int)
        self.ids = np.empty(0, dtype=np.int64)
        self._next_id = 0
        self.deposited = np.zeros(len(self.w))

    def add(self, x, k, m, z=None):
        """Add particles at the ceiling (or at depth ``z``)."""
        n = len(x)
        self.x = np.concatenate([self.x, x])
        self.z = np.concatenate([self.z, np.zeros(n) if z is None else z])
        self.m = np.concatenate([self.m, np.full(n, m) if np.isscalar(m)
                                 else m])
        self.k = np.concatenate([self.k, np.full(n, k, dtype=int)
                                 if np.isscalar(k) else k])
        self.ids = np.concatenate([self.ids, self._next_id
                                   + np.arange(n, dtype=np.int64)])
        self._next_id += n

    def step(self, dt):
        """Advance all particles by ``dt`` seconds."""
        n = self.x.size
        if n == 0:
            return
        s = np.sqrt(2 * self.kappa * dt)
        self.x += self.u * dt + s * self.rng.standard_normal(n)
        self.z += self.w[self.k] * dt + s * self.rng.standard_normal(n)
        self.z = np.abs(self.z)
        out = self.z >= self.prm.h_passage
        if out.any():
            np.add.at(self.deposited, self.k[out], self.m[out])
            keep = ~out
            self.x, self.z = self.x[keep], self.z[keep]
            self.m, self.k = self.m[keep], self.k[keep]
            self.ids = self.ids[keep]

    def concentration(self, xe, ze, nclass):
        """Concentration per class (g m^-3) on bins ``xe`` x ``ze``."""
        vol = self.prm.width * np.diff(xe)[None, :] * np.diff(ze)[:, None]
        out = np.zeros((nclass, len(ze) - 1, len(xe) - 1))
        for k in range(nclass):
            s = self.k == k
            out[k] = np.histogram2d(self.z[s], self.x[s], bins=(ze, xe),
                                    weights=self.m[s])[0] / vol
        return out


def fallout(prm, rel_t, rel_x, rel_m, d_um, t_out, rng, dt=1.0,
            n_target=20000, kappa=None, u=None, xe=None, ze=None,
            record=None):
    """Settle released silt and return concentration snapshots.

    Released mass is split among size classes by ``size_fractions`` and
    represented by particles of equal mass per class (stochastic rounding,
    so the expected mass is exact).  Returns the times, the class
    concentrations at each time in ``t_out`` on bins ``xe`` x ``ze``,
    and the Column object.  ``record(col, t)`` is called at each output.
    """
    d = np.asarray(d_um) * 1e-6
    frac = size_fractions(prm, d_um)
    col = Column(prm, stokes(prm, d), rng, kappa=kappa, u=u)
    xe = np.linspace(0, prm.length, 129) if xe is None else xe
    ze = np.linspace(0, prm.h_passage, 41) if ze is None else ze
    total = float(np.sum(rel_m))
    q = total * frac / n_target
    order = np.argsort(rel_t)
    rel_t, rel_x, rel_m = rel_t[order], rel_x[order], rel_m[order]
    t_out = np.sort(np.asarray(t_out, float))
    out = np.zeros((len(t_out), len(d), len(ze) - 1, len(xe) - 1))
    t, i, j = 0.0, 0, 0
    while j < len(t_out):
        t_next = t + dt
        i1 = np.searchsorted(rel_t, t_next, side="left")
        if i1 > i:
            for k in range(len(d)):
                lam = rel_m[i:i1] * frac[k] / q[k]
                nk = np.floor(lam + rng.random(lam.size)).astype(int)
                if nk.sum():
                    xs = np.repeat(rel_x[i:i1], nk)
                    xs = xs + prm.a * (rng.random(xs.size) - 0.5)
                    col.add(xs, k, q[k])
            i = i1
        col.step(dt)
        t = t_next
        while j < len(t_out) and t_out[j] <= t + 1e-9:
            out[j] = col.concentration(xe, ze, len(d))
            if record is not None:
                record(col, t)
            j += 1
    return t_out, out, col, xe, ze


def beam_attenuation(prm, conc, b_m):
    """Beam attenuation c (m^-1); class on the third-to-last axis."""
    return 4.8 / prm.vis_clear + np.einsum("k,...kij->...ij", b_m, conc)


def sighting_along(prm, c_line, dx, backward=True):
    """Sighting range along the passage from every bin of ``c_line``.

    The contrast of a black target decays as exp(-int c ds), so the
    range is the distance at which the optical depth reaches 4.8,
    Zaneveld and Pegau's criterion applied along a non-uniform path.
    Beyond the modeled section the water is clear.
    """
    c_w = 4.8 / prm.vis_clear
    c = np.asarray(c_line, float)
    if backward:
        c = c[..., ::-1]
    n = c.shape[-1]
    out = np.empty(c.shape)
    cum = np.concatenate([np.zeros(c.shape[:-1] + (1,)),
                          np.cumsum(c * dx, axis=-1)], axis=-1)
    for i in range(n):
        tau = cum[..., i:] - cum[..., i:i + 1]
        dist = np.arange(n + 1 - i) * dx
        rng_ = np.empty(c.shape[:-1])
        flat_tau = tau.reshape(-1, tau.shape[-1])
        flat_out = rng_.reshape(-1)
        for r, row in enumerate(flat_tau):
            k = np.searchsorted(row, 4.8)
            if k >= row.size:
                flat_out[r] = dist[-1] + (4.8 - row[-1]) / c_w
            else:
                f = (4.8 - row[k - 1]) / (row[k] - row[k - 1])
                flat_out[r] = dist[k - 1] + f * dx
        out[..., i] = rng_
    return out[..., ::-1] if backward else out


def sighting_range(prm, conc, b_m):
    """Black-target sighting range 4.8/c (m).

    ``conc`` has the size class on its third-to-last axis.
    """
    c_w = 4.8 / prm.vis_clear
    c = c_w + np.einsum("k,...kij->...ij", b_m, conc)
    return 4.8 / c
