"""Draw docs/media/tth_map.png: |q| and 2θ of every pixel of one flat detector at two arm tilts.

uv run --with matplotlib docs/media/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qorageom import tth_map, tth_to_q, wavelength

ROOT = Path(__file__).resolve().parents[2]
SHAPE = 1000, 1000
PIXEL_M = 75e-6
SDD_M = 0.15
ENERGY_KEV = 8.04
TILTS = 0.0, 25.0
LABEL_U_MM, LABEL_MARGIN_MM = 15.0, 4.0

lam = wavelength(ENERGY_KEV)
yc, xc = SHAPE[0] / 2, SHAPE[1] / 2
half_mm = 0.5 * SHAPE[0] * PIXEL_M * 1e3
extent = (-half_mm, half_mm, -half_mm, half_mm)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=150, layout="constrained")
maps = [tth_map(SHAPE, PIXEL_M, SDD_M, xc, yc, alpha_deg=a) for a in TILTS]
q_max = max(tth_to_q(m, lam).max() for m in maps)
for ax, tilt, tth in zip(axes, TILTS, maps, strict=True):
    image = ax.imshow(
        tth_to_q(tth, lam), extent=extent, cmap="viridis", vmin=0, vmax=q_max, origin="upper"
    )
    levels = np.arange(5, 60, 5)
    lines = ax.contour(tth[::-1], levels=levels, extent=extent, colors="white", linewidths=0.6)
    # matplotlib's own label placement clips labels at the frame, so pick one point per contour
    spots = []
    for segments in lines.allsegs:
        points = np.concatenate([s for s in segments if len(s)] or [np.empty((0, 2))])
        points = points[np.all(np.abs(points) < half_mm - LABEL_MARGIN_MM, axis=1)]
        if len(points):
            spots.append(points[np.abs(points[:, 0] - LABEL_U_MM).argmin()])
    ax.clabel(lines, fmt="%g°", fontsize=7, manual=spots)
    ax.set_title(f"arm tilted by {tilt:g}°", fontsize=9)
    ax.set_xlabel("u (mm)")
axes[0].set_ylabel("v (mm), up is +2θ")
fig.colorbar(image, ax=axes, label="|q| (Å$^{-1}$)", shrink=0.9)
fig.suptitle(
    f"tth_map: {SHAPE[0]} x {SHAPE[1]} pixels of {PIXEL_M * 1e6:g} µm, "
    f"{SDD_M * 1e3:g} mm from the sample, {ENERGY_KEV} keV; contours are 2θ",
    fontsize=9,
)
fig.savefig(ROOT / "docs/media/tth_map.png")
