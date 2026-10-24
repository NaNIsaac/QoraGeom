"""From a SPEC scan to a lab-frame volume in one call.

uv run python examples/spec_to_lab.py
"""

import tempfile
from pathlib import Path

import numpy as np

from qorageom.adapters.spec import Spec, geometry_from_spec

SPEC = """\
#F scan.spec
#O0 Delta Gamma Theta Chi Phi Energy camdist

#S 1 ascan Phi 0 0.5 50 1
#P0 37.0 -1.0 -71.35 85.26 12.0 9.0 500.0
#UIMDET 34idcTIM2:
#L Phi
0.0
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scan.spec"
        path.write_text(SPEC)
        scan = Spec(path).get_scan(1)

    geom = geometry_from_spec(scan, shape=(64, 64, 32))
    volume = np.random.default_rng(0).random(geom.shape)  # stand-in for a reconstructed object
    lab = geom.resample(volume)

    print(geom.axis, geom.detector, f"{geom.energy_kev} keV", f"{geom.distance_m} m")
    print(f"voxel {geom.voxel_size * 1e9:.2f} nm, cond {geom.diagnostics['cond']:.2f}")
    print(lab.shape)


if __name__ == "__main__":
    main()
