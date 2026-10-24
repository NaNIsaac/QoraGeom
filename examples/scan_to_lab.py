"""From the circle readings of a rocking scan to a lab-frame volume in one call.

uv run python examples/scan_to_lab.py
"""

import numpy as np

from qorageom import rocking_scan_geometry


def main():
    geom = rocking_scan_geometry(
        (64, 64, 32),
        energy_kev=9.0,
        distance_m=0.5,
        sample_angles_deg={"th": -71.35, "chi": 85.26, "phi": 12.0},
        detector_angles_deg={"delta": 37.0, "gamma": -1.0},
        rock="phi",
        step=0.01,
        diffractometer="34idc",
        detector="34idcTIM2",
    )
    volume = np.random.default_rng(0).random(geom.shape)  # stand-in for a reconstructed object
    lab = geom.resample(volume)

    print(geom.axis, geom.detector, f"{geom.energy_kev} keV", f"{geom.distance_m} m")
    print(f"voxel {geom.voxel_size * 1e9:.2f} nm, cond {geom.diagnostics['cond']:.2f}")
    print(lab.shape)


if __name__ == "__main__":
    main()
