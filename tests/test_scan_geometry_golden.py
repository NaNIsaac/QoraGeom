"""Golden values of geometry_from_spec, so that moving its assembly out of the SPEC path is checked.

``data/scan_geometry_golden.json`` was written by this file at QoraGeom d89f98e
(``uv run python tests/test_scan_geometry_golden.py``). The cases cover every rocking axis,
override, binning and diffractometer the existing tests use. Regenerate only for a change that is
meant to alter the geometry, and say so in the PR.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from qorageom.adapters.spec import Spec, geometry_from_spec

GOLDEN = Path(__file__).with_name("data") / "scan_geometry_golden.json"
SHAPE = (32, 32, 16)
NAMES = "Delta Gamma Theta Chi Phi Energy camdist"
POSITIONS = "37.0 -1.0 -71.35 85.26 12.0 9.0 500.0"
POSITIONS_ENERGY = "37.0 -1.0 -71.35 85.26 12.0 9000.0 500.0"
COMMANDS = {
    "phi": "ascan Phi 0 0.5 50 1",
    "theta": "ascan Theta -71.35 -71.30 50 1",
    "energy": "ascan en 9000 9003 30 1",
    "mesh": "mesh Phi 0 1 2 Chi 0 1 2 1",
}
ID01 = {"mu": 0.5, "eta": 20.0, "phi": 3.0, "nu": 0.7, "delta": 30.0}
P10 = {"mu": 0.5, "om": 20.0, "chi": 1.0, "phi": 3.0, "gam": 0.7, "del": 30.0}
CASES = {
    "phi": ("phi", {}),
    "theta": ("theta", {}),
    "energy": ("energy", {}),
    "overrides": ("phi", {"delta": 40.0, "energy": 9.5, "distance_m": 1.0}),
    "binning": ("phi", {"binning": (2, 4, 3)}),
    "scanmot": ("mesh", {"scanmot": "Phi", "scanmot_del": 0.01}),
    "chi_offset": ("phi", {"chi_offset": 80.0}),
    "tim1": ("phi", {"detector": "34idcTIM1"}),
    "id01": ("phi", {"diffractometer": "id01", "scanmot": "eta", "scanmot_del": 0.01, **ID01}),
    "p10": ("phi", {"diffractometer": "P10sixc", "scanmot": "om", "scanmot_del": 0.01, **P10}),
}


def scan(directory: Path, kind: str):
    positions = POSITIONS_ENERGY if kind == "energy" else POSITIONS
    path = directory / f"{kind}.spec"
    path.write_text(
        f"#F {kind}.spec\n#O0 {NAMES}\n\n#S 1 {COMMANDS[kind]}\n#P0 {positions}\n"
        "#UIMDET 34idcTIM2:\n#L a\n1\n",
        encoding="utf-8",
    )
    return Spec(path).get_scan(1)


def values(geom) -> dict:
    return {
        "diffractometer": geom.diffractometer,
        "detector": geom.detector,
        "axis": geom.axis,
        "shape": list(geom.shape),
        "wavelength_m": geom.wavelength_m,
        "energy_kev": geom.energy_kev,
        "distance_m": geom.distance_m,
        "voxel_size": geom.voxel_size,
        "B": geom.B.tolist(),
        "A": geom.A.tolist(),
        "lengths": geom.diagnostics["lengths"].tolist(),
        "angles_deg": geom.diagnostics["angles_deg"].tolist(),
        "cond": geom.diagnostics["cond"],
    }


def compute(directory: Path, name: str) -> dict:
    kind, overrides = CASES[name]
    return values(geometry_from_spec(scan(directory, kind), shape=SHAPE, **overrides))


@pytest.mark.parametrize("name", list(CASES))
def test_geometry_from_spec_matches_golden(tmp_path, name):
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    got = compute(tmp_path, name)
    for key in ("diffractometer", "detector", "axis", "shape"):
        assert got[key] == expected[key], key
    # The rocking column is the difference R0 R1^T q0 - q0 for steps down to 1e-3 deg
    # (1.7e-5 rad), which loses about five digits, and A inverts B (cond up to 50 here): libm
    # differences between platforms reach 2.5e-12 relative (macOS CI). A real change is >= 1e-6.
    for key, value in expected.items():
        if isinstance(value, str) or key == "shape":
            continue
        want = np.asarray(value, dtype=float)
        np.testing.assert_allclose(
            got[key], want, rtol=1e-9, atol=1e-9 * np.abs(want).max(), err_msg=key
        )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        table = {name: compute(Path(tmp), name) for name in CASES}
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
