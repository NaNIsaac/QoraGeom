"""Golden values of the scan geometry, from before the SPEC reader left QoraGeom.

``data/scan_geometry_golden.json`` was written at QoraGeom d89f98e by ``geometry_from_spec`` on
small SPEC files (that function now lives in CXD's ``cxd.io.spec``, with the same golden file). Each
case below passes the values that function read from its file, computed with the same float
operations, to ``rocking_scan_geometry``; the results must stay equal. Regenerate only for a change
that is meant to alter the geometry, and say so in the PR.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from qorageom import rocking_scan_geometry

GOLDEN = Path(__file__).with_name("data") / "scan_geometry_golden.json"
SHAPE = (32, 32, 16)
SAMPLE = {"th": -71.35, "chi": 85.26, "phi": 12.0}
ARM = {"delta": 37.0, "gamma": -1.0}
DISTANCE = 500.0 / 1000  # camdist [mm] in the files
PHI_STEP = (0.5 - 0.0) / 50  # ascan Phi 0 0.5 50
CASES = {
    "phi": {"rock": "phi", "step": PHI_STEP},
    "theta": {"rock": "th", "step": (-71.30 - -71.35) / 50},
    # ascan en 9000 9003 30 with Energy = 9000 eV: both scaled by 1e-3 to keV
    "energy": {"rock": "energy", "step": (9003.0 - 9000.0) / 30 * 1e-3, "energy": 9000.0 * 1e-3},
    "overrides": {"rock": "phi", "step": 0.01, "energy": 9.5, "distance": 1.0, "delta": 40.0},
    "binning": {"rock": "phi", "step": PHI_STEP, "binning": (2, 4, 3)},
    "scanmot": {"rock": "phi", "step": 0.01},
    "chi_offset": {"rock": "phi", "step": PHI_STEP, "sample_offsets_deg": (0.0, 80.0, 0.0)},
    "tim1": {"rock": "phi", "step": PHI_STEP, "detector": "34idcTIM1"},
    "id01": {
        "rock": "eta",
        "step": 0.01,
        "diffractometer": "id01",
        "sample": {"mu": 0.5, "eta": 20.0, "phi": 3.0},
        "arm": {"nu": 0.7, "delta": 30.0},
    },
    "p10": {
        "rock": "om",
        "step": 0.01,
        "diffractometer": "P10sixc",
        "sample": {"mu": 0.5, "om": 20.0, "chi": 1.0, "phi": 3.0},
        "arm": {"gam": 0.7, "del": 30.0},
    },
}


def compute(case: dict):
    arm = case.get("arm", ARM)
    if "delta" in case:
        arm = {**arm, "delta": case["delta"]}
    return rocking_scan_geometry(
        SHAPE,
        case.get("energy", 9.0),
        case.get("distance", DISTANCE),
        case.get("sample", SAMPLE),
        arm,
        case["rock"],
        case["step"],
        diffractometer=case.get("diffractometer", "34idc"),
        detector=case.get("detector", "34idcTIM2"),
        binning=case.get("binning", (1, 1, 1)),
        sample_offsets_deg=case.get("sample_offsets_deg"),
    )


@pytest.mark.parametrize("name", list(CASES))
def test_scan_geometry_matches_golden(name):
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    geom = compute(CASES[name])
    assert [geom.diffractometer, geom.detector, geom.axis, list(geom.shape)] == [
        expected["diffractometer"],
        expected["detector"],
        expected["axis"],
        expected["shape"],
    ]
    got = {
        "wavelength_m": geom.wavelength_m,
        "energy_kev": geom.energy_kev,
        "distance_m": geom.distance_m,
        "voxel_size": geom.voxel_size,
        "B": geom.B,
        "A": geom.A,
        "lengths": geom.diagnostics["lengths"],
        "angles_deg": geom.diagnostics["angles_deg"],
        "cond": geom.diagnostics["cond"],
    }
    # The rocking column is the difference R0 R1^T q0 - q0 for steps down to 1e-3 deg
    # (1.7e-5 rad), which loses about five digits, and A inverts B (cond up to 50 here): libm
    # differences between platforms reach 2.5e-12 relative (macOS CI). A real change is >= 1e-6.
    for key, value in got.items():
        want = np.asarray(expected[key], dtype=float)
        np.testing.assert_allclose(
            value, want, rtol=1e-9, atol=1e-9 * np.abs(want).max(), err_msg=key
        )
