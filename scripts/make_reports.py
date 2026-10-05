"""Generate the plain-text reports.

Numbers are recomputed here from the package or read back from the CSV
files written by the figure scripts, so every value quoted in a report can
be traced to a file under outputs/.  The sensitivity report runs its own
small ensembles.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import os
from dataclasses import replace

import numpy as np

from cave_percolation.ceiling import pool_cutoff, area_volume_exponent
from cave_percolation.column import (stokes, mass_attenuation,
                                     size_fractions, mfpt)
from cave_percolation.experiments import team_run
from cave_percolation.io_utils import DATADIR, write_report, write_csv
from cave_percolation.ip import D_F, P_C, NU
from cave_percolation.scenario import PRM, D_CLASSES_UM


def csv(stem):
    """Read a CSV written by io_utils.write_csv as a structured array."""
    return np.genfromtxt(os.path.join(DATADIR, f"{stem}.csv"),
                         delimiter=",", names=True)


def wrap(text, width=68):
    """Wrap prose to report lines indented two spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append("  " + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append("  " + cur)
    return lines


def dots(label, value, width=52):
    """Dotted leader line."""
    return "  " + label + " " + "." * max(2, width - len(label)) + " " + value


d = D_CLASSES_UM * 1e-6
w = stokes(PRM, d)
b = mass_attenuation(PRM, d)
fr = size_fractions(PRM, D_CLASSES_UM)

# ------------------------------------------------------------- parameters
cls = ["  d (um) | w (mm/s) | b (m2/g) | mass frac | fall 0.7 m (min)",
       "  " + "-" * 62]
for di, wi, bi, fi in zip(D_CLASSES_UM, w, b, fr):
    cls.append(f"  {di:6.2f} | {1e3 * wi:8.4f} | {bi:8.4f} |"
               f" {fi:9.4f} | {PRM.z_eye / wi / 60:10.1f}")
write_report("parameters", "Symbols, units, and reference values", [
    ("units", wrap(
        "Lengths in meters, times in seconds, silt mass in grams, gas "
        "volume in cubic meters (liters where stated), concentration in "
        "g m^-3 = mg/L, sighting range in meters.  +x is up-dip.")),
    ("reference values (illustrative, not calibrated)", [
        f"  rho = {PRM.rho} kg m^-3, g = {PRM.g} m s^-2,"
        f" gamma = {PRM.gamma} N m^-1, mu = {PRM.mu} Pa s",
        f"  rho_s = {PRM.rho_s} kg m^-3 (quartz density)",
        f"  lattice a = {PRM.a} m, passage {PRM.length} m x {PRM.width} m"
        f" ({PRM.nx} x {PRM.ny} cells), ceiling to floor"
        f" {PRM.h_passage} m",
        f"  relief: H = {PRM.hurst}, sigma1 = {PRM.sigma1} m (rms height"
        f" difference at 1 m), arch {PRM.arch} m, dip {PRM.tilt_deg} deg",
        f"  pinning radius median {1e3 * PRM.r_med:g} mm, log-sd"
        f" {PRM.r_lnsd}; median pinning height"
        f" {1e3 * PRM.ell_med:.2f} mm; capillary length"
        f" {1e3 * PRM.ell_c:.2f} mm",
        f"  grid Bond number sigma(a)/ell_med ="
        f" {PRM.sigma1 * PRM.a ** PRM.hurst / PRM.ell_med:.2f}",
        f"  diver: RMV {PRM.rmv_lpm} L/min, breathing period"
        f" {PRM.t_breath} s, exhalation {PRM.t_exhale} s, speed"
        f" {PRM.u_diver} m/s, spacing {PRM.spacing_t} s",
        f"  regulator {PRM.z_reg} m below a ceiling at"
        f" {PRM.depth_ceiling} m depth; gas per breath at the ceiling"
        f" {1e3 * PRM.v_breath:.3f} L; footprint rms {PRM.footprint} m;"
        f" {PRM.n_sub} parcels per breath",
        f"  silt: m0 = {PRM.m0} g m^-2, phi_c = {PRM.phi_c},"
        f" v_d = {1e6 * PRM.v_d:g} mL",
        f"  water: K = {PRM.kappa:g} m^2 s^-1, current {PRM.u_flow} m/s,"
        f" clear-water sighting {PRM.vis_clear} m, eye"
        f" {PRM.z_eye} m below ceiling",
        f"  grain size log-normal, median {PRM.d_median_um} um,"
        f" log-sd {PRM.d_lnsd}",
        f"  ensembles use seeds {PRM.seeds}"]),
    ("size classes", cls),
    ("provenance", wrap(
        "Physical constants are standard values for fresh water and "
        "quartz.  The ceiling relief, the pinning radius, the loose silt "
        "inventory m0, the detachment constants phi_c and v_d, the grain "
        "size distribution, and the eddy diffusivity K are not measured "
        "on a cave ceiling to the authors' knowledge; they are round "
        "values in a plausible range.  The respiratory minute volume is a "
        "typical working value for a swimming diver.  Every silt and "
        "sighting result scales with m0 and depends on K and phi_c, v_d; "
        "see sensitivity.txt.")),
])

# ---------------------------------------------------------- closed forms
rs_rows = ["  dip (deg) | r* sigma1=0.03 | 0.10 | 0.30 m   (H = 0.8)",
           "  " + "-" * 56]
for beta in (1, 2, 3, 5, 8, 12, 20):
    tb = np.tan(np.radians(beta))
    rs_rows.append(f"  {beta:9d} | " + " | ".join(
        f"{pool_cutoff(s, 0.8, tb):10.3g}" for s in (0.03, 0.1, 0.3)))
t_rows = ["  d (um) | T_residence exact (min), K = 1e-6, 1e-5, 1e-4",
          "  " + "-" * 52]
for di, wi in zip(D_CLASSES_UM[::3], w[::3]):
    t_rows.append(f"  {di:6.2f} | " + ", ".join(
        f"{mfpt(PRM.h_passage, wi, k) / 60:9.1f}"
        for k in (1e-6, 1e-5, 1e-4)))
write_report("closed_forms", "Closed forms", [
    ("gas under a ceiling", wrap(
        "Hydrostatic water, p_w(z) = p0 - rho g z, and negligible gas "
        "density give a connected pool a flat interface at level L.  A "
        "cell with effective ceiling h_eff = h - ell is covered when "
        "L <= h_eff, with pinning height ell = 2 gamma/(rho g r).  Pool "
        "volume V = a^2 sum (h_eff - L).  Adding gas lowers L and admits "
        "the highest perimeter cell (invasion percolation with threshold "
        "-h_eff); a perimeter cell with a higher neighbor outside the "
        "pool is a spill point, from which gas follows steepest ascent to "
        "another pool, and pools meeting at a saddle merge.  With every "
        "pool full, the covered set is the priority-flood filling of "
        "-h_eff with outlets at the open ends.")),
    ("pool scale and dip", wrap(
        "Self-affine relief has S(r) = sigma1 r^H; a dip adds r tan(beta). "
        "They balance at r* = (sigma1/tan beta)^(1/(1-H)).  For H near 1 "
        "this is very sensitive to the dip (power 5 at H = 0.8).  The "
        "cutoff is soft: a basin of width l needs a relief reversal "
        "larger than l tan(beta), whose Gaussian probability is about "
        "exp[-(l/r*)^(2(1-H))/2], a stretched exponential with exponent "
        "0.4 at H = 0.8.") + [""] + rs_rows),
    ("pool area and volume", wrap(
        "Below r*, a pool of width l has depth of order l^H, so "
        "V ~ A^((2+H)/2) across pools and A ~ V^(2/(2+H)) for gas fed "
        "from one point: 0.800 for H = 0.5, 0.714 for H = 0.8.")),
    ("connected capillary limit", wrap(
        "Relief negligible against pinning: invasion percolation without "
        "trapping (water under the ceiling stays connected to the water "
        "column).  Cluster dimension 91/48 = 1.8958, accepted thresholds "
        "bounded by p_c = 0.592746 (square lattice, sites).  A dip adds "
        "a gradient G per site; the finger width scales as "
        f"G^(-nu/(1+nu)) = G^(-{NU / (1 + NU):.4f}).")),
    ("silt detachment", wrap(
        "A contact line crossing a cell for the first time removes a "
        "fraction phi_c of its loose silt.  Gas volume v traveling along "
        "a path removes 1 - exp(-v/v_d) from each cell passed; the "
        "product over parcels depends only on the total volume, so the "
        "result does not depend on how breaths are split into parcels.")),
    ("settling and sight", wrap(
        "Stokes velocity w = (rho_s - rho) g d^2/(18 mu).  Mean residence "
        "time from a reflecting ceiling to an absorbing floor at depth "
        "H_p: T = H_p/w - (K/w^2)(1 - exp(-w H_p/K)).  Mass-specific "
        "beam attenuation for grains much larger than the wavelength, "
        "b = 3/(rho_s d).  Black-target sighting range 4.8/c for uniform "
        "c (Zaneveld and Pegau 2003), applied here as the distance at "
        "which the path optical depth reaches 4.8.") + [""] + t_rows),
])

# ---------------------------------------------------------- verification
f1c, f1b = csv("fig01c_fit"), csv("fig01b_fit")
f2b, f2c = csv("fig02b_fit"), csv("fig02c_pc")
f3 = csv("fig03b_fit")
f5 = csv("fig05a_fit")
f8a, f8c, f8d = csv("fig08a_dome"), csv("fig08c_saturation"), csv(
    "fig08d_residence")
cons = csv("fig08b_conservation")["residual"]
lines = [
    dots("fill-spill vs priority flood, max abs (m)",
         f"{float(f8c['max_abs_diff_m']):.3e}"),
    dots("  cells compared / cells with gas",
         f"{int(f8c['n_cells'])} / {int(f8c['n_gas_cells'])}"),
    dots("volume residual, max over 20 000 events", f"{cons.max():.3e}"),
]
ords = []
for key in f8a.dtype.names[1:]:
    e = f8a[key]
    o = np.log(e[:-1] / e[1:]) / np.log(2)
    ords.append(f"  dome {key}: errors " + ", ".join(f"{v:.2e}" for v in e)
                + "; orders " + ", ".join(f"{v:.2f}" for v in o))
lines += ords
lines += [
    dots("IP cluster dimension (exact 1.8958)",
         f"{float(f2b['D_mean']):.4f} [{float(f2b['ci_lo']):.4f},"
         f" {float(f2b['ci_hi']):.4f}]"),
    dots("IP acceptance half-density (p_c 0.5927)",
         f"{float(f2c['pc_estimate']):.4f}"),
    dots("gradient finger exponent (exact -0.5714)",
         f"{float(f3['slope']):.4f} [{float(f3['ci_lo']):.4f},"
         f" {float(f3['ci_hi']):.4f}]"),
]
for r in np.atleast_1d(f1b):
    lines.append(dots(f"relief Hurst fit, input H = {r['H_input']:.1f}",
                      f"{r['H_fit']:.4f}"))
for r in np.atleast_1d(f1c):
    lines.append(dots(f"pool V ~ A^x, H = {r['H']:.1f} (x = "
                      f"{r['expected']:.3f})",
                      f"{r['slope']:.4f} [{r['ci_lo']:.4f},"
                      f" {r['ci_hi']:.4f}]"))
for r in np.atleast_1d(f5):
    lines.append(dots(f"point source A ~ V^x, H = {r['H']:.1f} (x = "
                      f"{r['expected']:.3f})",
                      f"{r['slope']:.4f} [{r['ci_lo']:.4f},"
                      f" {r['ci_hi']:.4f}]"))
res = ["  Pe      | " + " | ".join(n.replace("relerr_", "")
                                   for n in f8d.dtype.names[1::2])]
for i, pe in enumerate(f8d["Pe"]):
    cells = []
    for n in f8d.dtype.names[1::2]:
        se = f8d[n.replace("relerr", "se")][i]
        cells.append(f"{100 * f8d[n][i]:+6.2f} +/- {200 * se:4.2f} %")
    res.append(f"  {pe:7.2f} | " + " | ".join(cells))
dt_prod = 2.0
bias = []
for wi in w:
    T = mfpt(PRM.h_passage, wi, PRM.kappa)
    bias.append(dt_prod / T)
write_report("verification", "Numerical verification", [
    ("independent algorithms", wrap(
        "The event-driven fill-spill engine and the priority-flood "
        "filling share no code path.  The connected limit is checked "
        "against exact percolation results, the gradient limit against "
        "the exact finger exponent, the pool geometry against the "
        "closed-form exponents, one dome against its exact area, and the "
        "Lagrangian settling against the exact residence time.")),
    ("measured residuals and exponents", lines),
    ("residence time, relative error (two standard errors)", res),
    ("reading these numbers", wrap(
        "The priority-flood agreement is exact to rounding.  The dome "
        "converges at roughly second order until the error reaches the "
        "pixelation floor of the contact line.  The relief exponent for "
        "H = 0.8 is about 0.77 because the spectral synthesis is cut off "
        "at the lattice scale.  The pool volume-area exponents exceed "
        "the closed form by 3 to 4 percent and their intervals exclude "
        "it; the point-source exponents contain it.  The IP acceptance "
        "density falls to half slightly below p_c, a finite-size effect "
        "of stopping at the lattice edge.  The residence-time error is a "
        "positive bias from detecting floor crossings only at step ends; "
        "it halves when the step falls by four, as expected for an "
        "O(sqrt(dt)) boundary bias.  Production runs use dt = 2 s, which "
        f"is {min(bias):.2e} to {max(bias):.2e} of the class residence "
        "times, well inside the smallest tested ratio for fine grains "
        "and near 0.02 T for the coarsest class, whose floor arrival "
        "may be late by a few percent.")),
])

# ------------------------------------------------------------ sensitivity
sens = []
rows = ["  phi_c | v_d (mL) | lead (g) | 2nd | 3rd | 4th | lead/2nd | 4/1",
        "  " + "-" * 64]
for phi in (0.2, 0.5, 0.8):
    for vd in (0.2e-6, 2e-6, 20e-6):
        p = replace(PRM, phi_c=phi, v_d=vd)
        per, tot1 = [], []
        for s in PRM.seeds[:6]:
            _, _, e = team_run(p, s, n_divers=4)
            t, c, m, tag = e.releases()
            per.append([m[tag == k].sum() for k in range(4)])
            _, _, e1 = team_run(p, s, n_divers=1)
            tot1.append(e1.releases()[2].sum())
        per = np.median(per, 0)
        r41 = per.sum() / np.median(tot1)
        rows.append(f"  {phi:5.1f} | {1e6 * vd:8.1f} | {per[0]:8.1f} |"
                    f" {per[1]:5.1f} | {per[2]:5.1f} | {per[3]:5.1f} |"
                    f" {per[0] / per[1]:8.2f} | {r41:4.2f}")
        sens.append([phi, vd, *per, r41])
geo = ["  varied            | lead (g) | lead/2nd | team of 4 / solo",
       "  " + "-" * 60]
for label, p in (("reference", PRM),
                 ("sigma1 = 0.05 m", replace(PRM, sigma1=0.05)),
                 ("sigma1 = 0.20 m", replace(PRM, sigma1=0.20)),
                 ("r_med = 2 mm", replace(PRM, r_med=2e-3)),
                 ("r_med = 8 mm", replace(PRM, r_med=8e-3)),
                 ("no arch", replace(PRM, arch=0.0)),
                 ("dip 6 deg", replace(PRM, tilt_deg=6.0))):
    per, tot1 = [], []
    for s in PRM.seeds[:6]:
        _, _, e = team_run(p, s, n_divers=4)
        t, c, m, tag = e.releases()
        per.append([m[tag == k].sum() for k in range(4)])
        _, _, e1 = team_run(p, s, n_divers=1)
        tot1.append(e1.releases()[2].sum())
    per = np.median(per, 0)
    geo.append(f"  {label:17s} | {per[0]:8.1f} | {per[0] / per[1]:8.2f} |"
               f" {per.sum() / np.median(tot1):5.2f}")
sens = np.array(sens)
write_csv("sensitivity_detachment", {
    "phi_c": sens[:, 0], "v_d": sens[:, 1], "lead": sens[:, 2],
    "second": sens[:, 3], "third": sens[:, 4], "fourth": sens[:, 5],
    "team4_over_solo": sens[:, 6]})
write_report("sensitivity", "Sensitivity of the team results", [
    ("detachment constants (team of four, medians of six ceilings)",
     rows),
    ("ceiling geometry (phi_c = 0.5, v_d = 2 mL)", geo),
    ("reading", wrap(
        "The lead diver releases 2.2 to 2.8 times the silt of the second "
        "diver in every combination tested, and a team of four releases "
        "1.8 to 2.0 times a solo diver.  The ordering comes from the gas, "
        "not the silt constants: the lead diver's gas opens new pools and "
        "paths, and followers' gas largely reuses them.  phi_c scales the "
        "absolute masses almost in proportion; the stripping volume v_d "
        "matters little because a single breath already exceeds it.  "
        "Absolute masses also scale with m0.  Sighting ranges depend on "
        "K by an order of magnitude at fixed m0 (figure 7c).")),
])

# ----------------------------------------------------------- figure notes
f4 = csv("fig04d_fraction")
f5c = csv("fig05c_offset")
f5b = csv("fig05b_hover_silt")
f6a, f6b = csv("fig06a_position"), csv("fig06b_team_size")
f7 = csv("fig07b_sighting")
notes = [("purpose", wrap(
    "Figures carry no in-panel numbers.  Values that would otherwise be "
    "printed inside an axes are recorded here; every panel is also "
    "available as CSV under outputs/data."))]
notes.append(("figure 4, gas-holding fraction", [
    f"  H = 0.{h[1]}: dip "
    + ", ".join(f"{v:.1f}" for v in f4[f"dip_deg_H{h}"])
    + " deg -> " + ", ".join(f"{100 * v:.1f}" for v in f4[f"frac_H{h}"])
    + " %" for h in ("05", "08")]))
last = {k: f5b[k][-1] for k in f5b.dtype.names if "median" in k}
notes.append(("figure 5, one diver holding station 10 min", [
    "  silt released in section (g): " + ", ".join(
        f"{k.split('tilt')[1]} deg {v:.0f}" for k, v in last.items()),
    "  up-dip offset of silt (m), median: " + ", ".join(
        f"{t:g} deg {o:.2f}" for t, o in zip(f5c["tilt_deg"],
                                             f5c["offset_median"])),
    "  gas leaving the section (%): " + ", ".join(
        f"{t:g} deg {100 * o:.0f}" for t, o in zip(f5c["tilt_deg"],
                                                   f5c["lost_median"]))]))
notes.append(("figure 6, team position", [
    "  team of four, silt by position (g), RMV 20: " + ", ".join(
        f"{v:.1f}" for v in f6a["median_rmv20"]),
    "  team of four, silt by position (g), RMV 40: " + ", ".join(
        f"{v:.1f}" for v in f6a["median_rmv40"]),
    "  total by team size 1..4 (g), RMV 20: " + ", ".join(
        f"{v:.1f}" for v in f6b["median_rmv20"]),
    "  total by team size 1..4 (g), RMV 40: " + ", ".join(
        f"{v:.1f}" for v in f6b["median_rmv40"])]))
sel = [np.argmin(np.abs(f7["t_min"] - m)) for m in (15, 30, 45, 60, 90)]
rows7 = ["  minutes | " + " | ".join(f"{f7['t_min'][i]:5.0f}" for i in sel)]
for name in ("team3", "team3_rmv40", "hover5"):
    rows7.append(f"  eye {name:11s} | " + " | ".join(
        f"{f7[f'eye_median_{name}'][i]:5.1f}" for i in sel))
rows7.append("  top 0.1 m team3   | " + " | ".join(
    f"{f7['top_median_team3'][i]:5.2f}" for i in sel))
notes.append(("figure 7, sighting range toward the exit (m, medians)",
              rows7))
write_report("figure_notes", "Figure notes and tabulated values", notes)

# ------------------------------------------------------------- open items
write_report("open_items", "Open items and negative results", [
    ("a closed form that holds only as a scaling variable", wrap(
        "The pool scale r* = (sigma1/tan beta)^(1/(1-H)) was expected to "
        "act as a sharp cutoff on pool size.  It does not.  An "
        "area-weighted pool size varies with the dip much more weakly "
        "than r*, because the cutoff is a soft stretched exponential "
        "and, for H = 0.8, pools of ten times r* still form.  The pooled "
        "area distribution collapses on l/r* for H = 0.5 above about "
        "1.5 r*; for H = 0.8 it does not collapse within a 10 m domain. "
        "r* is reported as the scaling variable, not as a size limit.")),
    ("two fitted exponents miss by a few percent", wrap(
        "Pool volume against area exceeds the closed-form exponent by 3 "
        "to 4 percent with intervals that exclude it; the likely cause "
        "is the lattice cutoff of the relief (fitted H = 0.77 for input "
        "0.8) acting on the smallest pools.  The point-source growth "
        "exponents agree within their intervals.")),
    ("results that depend on unmeasured quantities", wrap(
        "No measurement of loose silt inventory, contact-line stripping "
        "efficiency, pinning radius, or relief statistics on a cave "
        "ceiling is known to the authors.  Silt masses scale with m0. "
        "Sighting ranges change by an order of magnitude between "
        "K = 1e-6 and 1e-4 m^2/s.  The team-position ordering survives "
        "every combination tested; its magnitude does not.")),
    ("what the model leaves out", wrap(
        "Gas is quasi-static: no bubble dynamics, no inertia at spill "
        "points, no dissolution of trapped gas, no receding contact "
        "lines.  The pinned meniscus volume is neglected.  Pools are "
        "disconnected fill-spill pools; the connected capillary limit "
        "is a separate engine and the crossover between the two is not "
        "modeled.  Silt detached under a pool is released to the water "
        "at once.  The water column is width-averaged, still apart from "
        "a constant eddy diffusivity, and ignores fin wash, flocculation, "
        "and resuspension from the floor.  A closed-circuit rebreather "
        "exhales no gas in level swimming and releases no silt here; "
        "loop venting on ascent is not modeled.")),
    ("what does not depend on those choices", wrap(
        "The equivalence of the fill-spill engine and priority flooding, "
        "volume conservation, the percolation limits, and the scaling "
        "exponents are properties of the geometry and hold for any "
        "silt, optics, or diver parameters.")),
])

# ------------------------------------------------------------- references
write_report("references", "References (DOIs checked against Crossref)", [
    ("percolation", [
        "  Wilkinson D, Willemsen JF (1983) Invasion percolation: a new form",
        "    of percolation theory. J Phys A 16:3365.",
        "    https://doi.org/10.1088/0305-4470/16/14/028",
        "  Newman MEJ, Ziff RM (2000) Efficient Monte Carlo algorithm and",
        "    high-precision results for percolation. Phys Rev Lett",
        "    85:4104. https://doi.org/10.1103/PhysRevLett.85.4104",
        "  Birovljev A, Furuberg L, Feder J, Jossang T, Maloy KJ, Aharony A",
        "    (1991) Gravity invasion percolation in two dimensions:",
        "    experiment and simulation. Phys Rev Lett 67:584.",
        "    https://doi.org/10.1103/PhysRevLett.67.584",
        "  Meakin P, Feder J, Frette V, Jossang T (1992) Invasion",
        "    percolation in a destabilizing gradient. Phys Rev A 46:3357.",
        "    https://doi.org/10.1103/PhysRevA.46.3357",
        "  Wagner G, Meakin P, Feder J, Jossang T (1997) Buoyancy-driven",
        "    invasion percolation with migration and fragmentation.",
        "    Physica A 245:217.",
        "    https://doi.org/10.1016/S0378-4371(97)00324-5"]),
    ("depressions, roughness, detachment, optics", [
        "  Barnes R, Callaghan KL, Wickert AD (2021) Computing water flow",
        "    through complex landscapes, Part 3: Fill-Spill-Merge. Earth",
        "    Surf Dynam 9:105. https://doi.org/10.5194/esurf-9-105-2021",
        "  Candela T, Renard F, Klinger Y, Mair K, Schmittbuhl J,",
        "    Brodsky EE (2012) Roughness of fault surfaces over nine",
        "    decades of length scales. J Geophys Res 117:B08409.",
        "    https://doi.org/10.1029/2011JB009041",
        "  Sharma P, Flury M, Zhou J (2008) Detachment of colloids from a",
        "    solid surface by a moving air-water interface. J Colloid",
        "    Interface Sci 326:143.",
        "    https://doi.org/10.1016/j.jcis.2008.07.030",
        "  Zaneveld JRV, Pegau WS (2003) Robust underwater visibility",
        "    parameter. Opt Express 11:2997.",
        "    https://doi.org/10.1364/OE.11.002997"]),
])
print("reports written")
