"""Dark style, color maps, shading, and looping GIF export.

Animations are rendered frame by frame with Matplotlib into RGB arrays and
written as palette GIFs with Pillow.  Every frame is a direct render of a
computed model field.  The only display operations are hillshading of the
ceiling relief (illumination only, no change to where anything is),
bilinear interpolation between bin centers of binned particle fields, and
the logarithmic color scales stated on each color bar.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from PIL import Image

matplotlib.use("Agg")

__all__ = ["dark", "ROCK", "GAS", "SILT", "TRAIL", "VIS", "BG", "FG",
           "TRAIL_NORM", "trail_layer", "hillshade", "fig_to_rgb",
           "write_gif", "ANIMDIR", "scalebar"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

BG = "#06080d"
FG = "#d9dee8"

# limestone seen by a dive light: cold shadow to warm lit rock
ROCK = LinearSegmentedColormap.from_list("rock", [
    (0.00, "#0b0c10"), (0.35, "#2b2a2c"), (0.65, "#6b6259"),
    (0.85, "#a89a86"), (1.00, "#d8cbb4")])

# trapped gas: the mercury mirror divers see on a ceiling
GAS = LinearSegmentedColormap.from_list("gas", [
    (0.00, "#3d5566"), (0.30, "#7fa3b8"), (0.60, "#c6dbe6"),
    (0.85, "#eef6fa"), (1.00, "#ffffff")])

# suspended silt: clear dark water to umber, ochre, and cream
SILT = LinearSegmentedColormap.from_list("silt", [
    (0.00, BG), (0.18, "#1d1812"), (0.40, "#5a3d1f"),
    (0.62, "#9c6b2e"), (0.82, "#d4a55c"), (1.00, "#f6e7c4")])

# migrating gas trails
TRAIL = LinearSegmentedColormap.from_list("trail", [
    (0.00, "#123040"), (0.50, "#3fb6d9"), (1.00, "#e9fbff")])

# gas volume that has migrated across a cell, liters, logarithmic
TRAIL_NORM = LogNorm(1e-3, 20.0)


def trail_layer(flux_m3, norm=TRAIL_NORM):
    """Opaque trail colors and a mask for cells crossed by migrating gas.

    ``flux_m3`` is the cumulative gas volume (m^3) that has passed each
    cell.  Colors follow TRAIL under ``norm`` (liters) exactly, with no
    blending, so they can be read against a color bar in liters.
    """
    liters = 1e3 * np.asarray(flux_m3)
    mask = liters > 0
    lv = np.clip(norm(np.where(mask, liters, norm.vmin)), 0.0, 1.0)
    return TRAIL(lv)[..., :3], mask


# sighting range, short (hazard) to long (clear)
VIS = LinearSegmentedColormap.from_list("vis", [
    (0.00, "#8a1c0a"), (0.30, "#e0741f"), (0.60, "#e8d27a"),
    (1.00, "#7fd3c4")])


def dark():
    """Apply the animation style."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#3a4152",
        "xtick.color": FG, "ytick.color": FG, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.6,
    })


def hillshade(h, a, azimuth=315.0, altitude=35.0, exag=6.0):
    """Lambertian hillshade in [0, 1] of an elevation field (m)."""
    gy, gx = np.gradient(h * exag, a)
    az, alt = np.radians(azimuth), np.radians(altitude)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    s = (np.sin(alt) * np.cos(slope)
         + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(s, 0.0, 1.0)


def scalebar(ax, x0, y0, length, label, color=FG):
    """Horizontal scale bar in data units."""
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=1.6,
            solid_capstyle="butt")
    ax.text(x0 + 0.5 * length, y0, label, color=color, ha="center",
            va="bottom", fontsize=8)


def fig_to_rgb(fig):
    """Rasterize a figure to an (h, w, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def write_gif(stem, frames, fps=18, colors=224):
    """Write a looping, palette-quantized GIF and return its path.

    One adaptive palette is computed from a sample of frames and shared
    by all of them, which removes frame-to-frame palette flicker.
    """
    os.makedirs(ANIMDIR, exist_ok=True)
    path = os.path.join(ANIMDIR, f"{stem}.gif")
    pick = frames[:: max(1, len(frames) // 12)]
    strip = Image.fromarray(np.concatenate(pick, axis=0))
    pal = strip.quantize(colors=colors, method=Image.MEDIANCUT,
                         dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=pal,
                                        dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(round(1000 / fps)), optimize=True, disposal=2)
    return path
