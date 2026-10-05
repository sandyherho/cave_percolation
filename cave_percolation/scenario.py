"""Reference parameters.

Every value here is illustrative.  The ceiling roughness, the loose silt
inventory, and the detachment constants have not been measured on a cave
ceiling to the authors' knowledge; the reports state which results depend
on them and which do not.  Lengths are in meters, times in seconds, masses
in grams, volumes in cubic meters unless a name says otherwise.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Params", "PRM", "D_CLASSES_UM"]

# particle size classes (micrometres), clay to fine sand, spaced by 2^(1/3)
# so that neighboring settling fronts overlap after an hour of settling
D_CLASSES_UM = 2.0 ** (np.arange(22) / 3.0)


@dataclass(frozen=True)
class Params:
    """Physical and numerical constants shared by all scripts."""

    # fluid and grain properties (fresh water, quartz-density grains)
    rho: float = 1000.0          # kg m^-3
    g: float = 9.81              # m s^-2
    gamma: float = 0.072         # N m^-1, air-water
    mu: float = 1.0e-3           # Pa s
    rho_s: float = 2650.0        # kg m^-3
    p_atm: float = 101325.0      # Pa

    # ceiling lattice and roughness
    a: float = 0.02              # lattice spacing (m)
    nx: int = 640                # along passage, +x is up-dip
    ny: int = 160                # across passage
    hurst: float = 0.8           # Hurst exponent of the relief
    sigma1: float = 0.10         # rms height difference at 1 m lag (m)
    tilt_deg: float = 2.0        # ceiling dip along the passage
    arch: float = 0.15           # center-to-wall ceiling drop (m)
    r_med: float = 4.0e-3        # median contact-line pinning radius (m)
    r_lnsd: float = 0.5          # log-standard deviation of that radius

    # divers (open circuit)
    rmv_lpm: float = 20.0        # respiratory minute volume at depth, L/min
    t_breath: float = 4.0        # breathing period (s)
    t_exhale: float = 1.6        # exhalation duration (s)
    u_diver: float = 0.20        # swimming speed (m s^-1)
    z_reg: float = 0.8           # regulator depth below the ceiling (m)
    depth_ceiling: float = 20.0  # water depth of the ceiling (m)
    footprint: float = 0.12      # rms radius of the bubble footprint (m)
    n_sub: int = 40              # gas parcels per breath
    spacing_t: float = 20.0      # time between divers in a team (s)

    # ceiling silt
    m0: float = 50.0             # loose silt inventory (g m^-2)
    phi_c: float = 0.5           # fraction removed by one contact-line pass
    v_d: float = 2.0e-6          # e-folding gas volume for path stripping

    # water column and optics
    h_passage: float = 2.0       # ceiling to floor (m)
    z_eye: float = 0.7           # diver eye depth below the ceiling (m)
    kappa: float = 1.0e-5        # eddy diffusivity (m^2 s^-1)
    u_flow: float = 0.0          # along-passage current (m s^-1)
    vis_clear: float = 30.0      # clear-water sighting range (m)
    d_median_um: float = 8.0     # median grain size of the loose silt
    d_lnsd: float = 1.0          # log-standard deviation of grain size

    seeds: tuple = field(default=(11, 23, 37, 41, 53, 67, 79, 83, 97, 101,
                                  113, 127))

    @property
    def tanb(self):
        """Tangent of the ceiling dip."""
        return float(np.tan(np.radians(self.tilt_deg)))

    @property
    def width(self):
        """Return the passage width in meters."""
        return self.ny * self.a

    @property
    def length(self):
        """Return the modeled passage length in meters."""
        return self.nx * self.a

    @property
    def ell_c(self):
        """Capillary length sqrt(gamma/(rho g)) (m)."""
        return float(np.sqrt(self.gamma / (self.rho * self.g)))

    @property
    def ell_med(self):
        """Median pinning height 2 gamma/(rho g r_med) (m)."""
        return 2.0 * self.gamma / (self.rho * self.g * self.r_med)

    @property
    def v_breath(self):
        """Exhaled volume per breath at ceiling pressure (m^3)."""
        v_amb = self.rmv_lpm * 1e-3 * self.t_breath / 60.0
        p_reg = self.p_atm + self.rho * self.g * (self.depth_ceiling
                                                  + self.z_reg)
        p_ceil = self.p_atm + self.rho * self.g * self.depth_ceiling
        return v_amb * p_reg / p_ceil


PRM = Params()
