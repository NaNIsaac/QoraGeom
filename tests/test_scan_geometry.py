import numpy as np
import pytest

from qorageom import (
    direct_basis,
    energy_basis,
    geometry_from_spec,
    reciprocal_basis,
    wavelength,
)
from qorageom.spec import Spec

HEADER = """\
#F geom.spec
#O0 Delta Gamma Theta Chi Phi Energy camdist
"""
SCANS = {
    "phi": ("ascan Phi 0 0.5 50 1", "37.0 -1.0 -71.35 85.26 12.0 9.0 500.0"),
    "theta": ("ascan Theta -71.35 -71.30 50 1", "37.0 -1.0 -71.35 85.26 12.0 9.0 500.0"),
    "energy": ("ascan en 9000 9003 30 1", "37.0 -1.0 -71.35 85.26 12.0 9000.0 500.0"),
    "mesh": ("mesh Phi 0 1 2 Chi 0 1 2 1", "37.0 -1.0 -71.35 85.26 12.0 9.0 500.0"),
}


def _spec(tmp_path, *, motors=None, detector="34idcTIM2:", scan="phi"):
    command, positions = SCANS[scan]
    names = "Delta Gamma Theta Chi Phi Energy camdist"
    if motors is not None:
        positions = " ".join(p for p, n in zip(positions.split(), names.split(), strict=True)
                             if n in motors)
        names = " ".join(motors)
    text = f"#F geom.spec\n#O0 {names}\n\n#S 1 {command}\n#P0 {positions}\n"
    if detector:
        text += f"#UIMDET {detector}\n"
    text += "#L a\n1\n"
    path = tmp_path / "geom.spec"
    path.write_text(text, encoding="utf-8")
    return Spec(path).get_scan(1)


SHAPE = (32, 32, 16)
LAM = wavelength(9.0) * 1e-10
ANGLES = {"gamma": np.radians(-1.0), "delta": np.radians(37.0)}


def test_spec_helpers(spec_file):
    scan = Spec(spec_file).get_scan(1)
    assert scan.detector_distance_m == pytest.approx(0.12345)  # from #UXML sdd [mm]
    assert scan.energy_kev is None


def test_spec_helpers_use_motors(tmp_path):
    scan = _spec(tmp_path, scan="energy")
    assert scan.energy_kev == pytest.approx(9.0)  # 9000 eV
    assert scan.detector_distance_m == pytest.approx(0.5)
    assert _spec(tmp_path).energy_kev == pytest.approx(9.0)


def test_phi_scan_matches_reciprocal_basis(tmp_path):
    geom = geometry_from_spec(_spec(tmp_path), shape=SHAPE)
    expected = reciprocal_basis(
        LAM, np.radians(0.01), 0.5, **ANGLES, pixel_size=55e-6,
        theta=np.radians(-71.35), chi=np.radians(85.26), axis="phi",
    )
    expected[:, 1] *= -1  # TIM2 axis 1 runs along y-
    np.testing.assert_allclose(geom.B, expected)
    np.testing.assert_allclose(geom.A, direct_basis(expected, SHAPE))
    assert (geom.axis, geom.detector, geom.shape) == ("phi", "34idcTIM2", SHAPE)
    assert geom.wavelength_m == pytest.approx(LAM)
    assert geom.diagnostics["cond"] > 1


def test_theta_scan_uses_theta_axis(tmp_path):
    geom = geometry_from_spec(_spec(tmp_path, scan="theta"), shape=SHAPE)
    expected = reciprocal_basis(LAM, np.radians(0.001), 0.5, **ANGLES, pixel_size=55e-6)
    np.testing.assert_allclose(geom.B[:, 2], expected[:, 2])


def test_energy_scan_uses_energy_basis(tmp_path):
    geom = geometry_from_spec(_spec(tmp_path, scan="energy"), shape=SHAPE)
    expected = energy_basis(9.0, 0.0001, 0.5, **ANGLES, pixel_size=55e-6)
    np.testing.assert_allclose(geom.B[:, 2], expected[:, 2])
    assert geom.axis == "energy"
    assert geom.energy_kev == pytest.approx(9.0)


def test_overrides_win_over_the_file(tmp_path):
    scan = _spec(tmp_path)
    base = geometry_from_spec(scan, shape=SHAPE)
    moved = geometry_from_spec(scan, shape=SHAPE, delta=40.0, energy=9.5, distance_m=1.0)
    assert not np.allclose(base.B, moved.B)
    assert moved.energy_kev == pytest.approx(9.5)
    assert moved.distance_m == 1.0
    expected = reciprocal_basis(
        wavelength(9.5) * 1e-10, np.radians(0.01), 1.0, np.radians(-1.0), np.radians(40.0),
        55e-6, theta=np.radians(-71.35), chi=np.radians(85.26), axis="phi",
    )
    np.testing.assert_allclose(moved.B[:, 2], expected[:, 2])


def test_binning_scales_pixels_and_step(tmp_path):
    scan = _spec(tmp_path)
    one = geometry_from_spec(scan, shape=SHAPE)
    binned = geometry_from_spec(scan, shape=SHAPE, binning=(2, 4, 3))
    np.testing.assert_allclose(binned.B[:, 0], 2 * one.B[:, 0])
    np.testing.assert_allclose(binned.B[:, 1], 4 * one.B[:, 1])
    assert np.linalg.norm(binned.B[:, 2]) == pytest.approx(3 * np.linalg.norm(one.B[:, 2]), rel=1e-3)


def test_scanmot_override_for_other_commands(tmp_path):
    scan = _spec(tmp_path, scan="mesh")
    geom = geometry_from_spec(scan, shape=SHAPE, scanmot="Phi", scanmot_del=0.01)
    assert geom.axis == "phi"


def test_resample_returns_lab_volume(tmp_path):
    geom = geometry_from_spec(_spec(tmp_path), shape=SHAPE)
    volume = np.random.default_rng(0).random(SHAPE)
    assert geom.resample(volume).shape == SHAPE


def test_missing_detector_names_the_argument(tmp_path):
    scan = _spec(tmp_path, detector="")
    with pytest.raises(ValueError, match="detector="):
        geometry_from_spec(scan, shape=SHAPE)
    assert geometry_from_spec(scan, detector="34idcTIM1", shape=SHAPE).detector == "34idcTIM1"


def test_missing_shape_names_the_argument(tmp_path):
    with pytest.raises(ValueError, match="shape="):
        geometry_from_spec(_spec(tmp_path))


def test_missing_energy_and_distance_name_the_override(tmp_path):
    with pytest.raises(ValueError, match="energy=") as err:
        geometry_from_spec(_spec(tmp_path, motors=["Delta", "Gamma", "Theta", "Chi", "Phi", "camdist"]),
                           shape=SHAPE)
    assert "scan 1" in str(err.value)
    with pytest.raises(ValueError, match="distance_m="):
        geometry_from_spec(_spec(tmp_path, motors=["Delta", "Gamma", "Theta", "Chi", "Phi", "Energy"]),
                           shape=SHAPE)


def test_missing_motor_and_scan_axis(tmp_path):
    scan = _spec(tmp_path, motors=["Delta", "Theta", "Chi", "Phi", "Energy", "camdist"])
    with pytest.raises(ValueError, match="gamma="):
        geometry_from_spec(scan, shape=SHAPE)
    with pytest.raises(ValueError, match="scanmot and scanmot_del"):
        geometry_from_spec(_spec(tmp_path, scan="mesh"), shape=SHAPE)


def test_rocking_about_chi_unsupported(tmp_path):
    with pytest.raises(ValueError, match="cannot rock"):
        geometry_from_spec(_spec(tmp_path, scan="mesh"), shape=SHAPE, scanmot="chi", scanmot_del=0.1)


def test_unknown_override_and_diffractometer(tmp_path):
    scan = _spec(tmp_path)
    with pytest.raises(TypeError, match="deltta"):
        geometry_from_spec(scan, shape=SHAPE, deltta=1.0)
    with pytest.raises(NotImplementedError):
        geometry_from_spec(scan, diffractometer="id01", shape=SHAPE)
