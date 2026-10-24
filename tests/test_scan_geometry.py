import numpy as np
import pytest

from qorageom import (
    direct_basis,
    energy_basis,
    reciprocal_basis,
    rocking_scan_geometry,
    wavelength,
)
from qorageom.adapters.spec import Spec, geometry_from_spec
from qorageom.detector import Detector

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
        positions = " ".join(
            p for p, n in zip(positions.split(), names.split(), strict=True) if n in motors
        )
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
        LAM,
        np.radians(0.01),
        0.5,
        **ANGLES,
        pixel_size=55e-6,
        theta=np.radians(-71.35),
        chi=np.radians(85.26),
        axis="phi",
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
    np.testing.assert_allclose(geom.B[:, 2], expected[:, 2], rtol=1e-7, atol=1e-6)


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
        wavelength(9.5) * 1e-10,
        np.radians(0.01),
        1.0,
        np.radians(-1.0),
        np.radians(40.0),
        55e-6,
        theta=np.radians(-71.35),
        chi=np.radians(85.26),
        axis="phi",
    )
    np.testing.assert_allclose(moved.B[:, 2], expected[:, 2])


def test_binning_scales_pixels_and_step(tmp_path):
    scan = _spec(tmp_path)
    one = geometry_from_spec(scan, shape=SHAPE)
    binned = geometry_from_spec(scan, shape=SHAPE, binning=(2, 4, 3))
    np.testing.assert_allclose(binned.B[:, 0], 2 * one.B[:, 0])
    np.testing.assert_allclose(binned.B[:, 1], 4 * one.B[:, 1])
    assert np.linalg.norm(binned.B[:, 2]) == pytest.approx(
        3 * np.linalg.norm(one.B[:, 2]), rel=1e-3
    )


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
        geometry_from_spec(
            _spec(tmp_path, motors=["Delta", "Gamma", "Theta", "Chi", "Phi", "camdist"]),
            shape=SHAPE,
        )
    assert "scan 1" in str(err.value)
    with pytest.raises(ValueError, match="distance_m="):
        geometry_from_spec(
            _spec(tmp_path, motors=["Delta", "Gamma", "Theta", "Chi", "Phi", "Energy"]), shape=SHAPE
        )


def test_missing_motor_and_scan_axis(tmp_path):
    scan = _spec(tmp_path, motors=["Delta", "Theta", "Chi", "Phi", "Energy", "camdist"])
    with pytest.raises(ValueError, match="gamma="):
        geometry_from_spec(scan, shape=SHAPE)
    with pytest.raises(ValueError, match="scanmot and scanmot_del"):
        geometry_from_spec(_spec(tmp_path, scan="mesh"), shape=SHAPE)


def test_rocking_about_chi_and_a_detector_motor(tmp_path):
    scan = _spec(tmp_path, scan="mesh")
    assert geometry_from_spec(scan, shape=SHAPE, scanmot="Chi", scanmot_del=0.01).axis == "chi"
    with pytest.raises(ValueError, match="cannot rock"):
        geometry_from_spec(scan, shape=SHAPE, scanmot="delta", scanmot_del=0.1)


def test_unknown_override_and_diffractometer(tmp_path):
    scan = _spec(tmp_path)
    with pytest.raises(TypeError, match="deltta"):
        geometry_from_spec(scan, shape=SHAPE, deltta=1.0)
    with pytest.raises(ValueError, match="mu="):  # the 34-ID-C file has no mu motor
        geometry_from_spec(scan, diffractometer="id01", shape=SHAPE)


def test_id01_and_p10_geometry_from_overrides(tmp_path):
    scan = _spec(tmp_path)
    angles = {"mu": 0.5, "eta": 20.0, "phi": 3.0, "nu": 0.7, "delta": 30.0}
    id01 = geometry_from_spec(
        scan, diffractometer="id01", shape=SHAPE, scanmot="eta", scanmot_del=0.01, **angles
    )
    assert id01.axis == "eta"
    p10 = geometry_from_spec(
        scan,
        diffractometer="P10sixc",
        shape=SHAPE,
        scanmot="om",
        scanmot_del=0.01,
        mu=0.5,
        om=20.0,
        chi=1.0,
        phi=3.0,
        gam=0.7,
        **{"del": 30.0},
    )
    assert p10.axis == "om"
    assert np.all(np.isfinite(p10.B)) and np.all(np.isfinite(id01.B))


def test_sample_offset_override(tmp_path):
    scan = _spec(tmp_path)
    shifted = geometry_from_spec(scan, shape=SHAPE, chi_offset=80.0)
    assert not np.allclose(shifted.B, geometry_from_spec(scan, shape=SHAPE).B)


def test_plain_arguments_give_the_spec_result(tmp_path):
    spec = geometry_from_spec(_spec(tmp_path), shape=SHAPE, binning=(2, 4, 3))
    plain = rocking_scan_geometry(
        SHAPE,
        9.0,
        0.5,
        {"th": -71.35, "chi": 85.26, "phi": 12.0},
        [37.0, -1.0],
        "phi",
        0.01,
        diffractometer="34idc",
        detector="34idcTIM2",
        binning=(2, 4, 3),
    )
    for name in ("B", "A"):
        assert np.array_equal(getattr(plain, name), getattr(spec, name))
    assert plain.voxel_size == spec.voxel_size
    assert (plain.axis, plain.detector, plain.shape) == (spec.axis, spec.detector, spec.shape)


def test_plain_energy_scan_takes_kev_steps(tmp_path):
    spec = geometry_from_spec(_spec(tmp_path, scan="energy"), shape=SHAPE)  # eV in the file
    plain = rocking_scan_geometry(
        SHAPE,
        9.0,
        0.5,
        [-71.35, 85.26, 12.0],
        [37.0, -1.0],
        "energy",
        0.0001,
        diffractometer="34idc",
        detector="34idcTIM2",
    )
    np.testing.assert_allclose(plain.B, spec.B, rtol=1e-12)
    assert plain.axis == "energy"


def test_plain_arguments_accept_a_custom_detector():
    det = Detector("square", (100e-6, 100e-6), ("x+", "y-"), (128, 128))
    geom = rocking_scan_geometry(
        SHAPE,
        9.0,
        1.0,
        [0.0, 90.0, 0.0],
        [30.0, 0.0],
        "th",
        0.01,
        diffractometer="34idc",
        detector=det,
    )
    assert geom.detector == "square"
    assert np.linalg.norm(geom.B[:, 0]) == pytest.approx(100e-6 / (LAM * 1.0))


@pytest.mark.parametrize(
    ("sample", "rock", "match"),
    [
        ({"th": 0.0, "chi": 90.0}, "th", "missing"),
        ({"th": 0.0, "chi": 90.0, "phi": 0.0, "mu": 1.0}, "th", "unknown"),
        ([0.0, 90.0], "th", "3 values"),
        ([0.0, 90.0, 0.0], "Phi", "cannot rock"),
    ],
)
def test_plain_arguments_reject_bad_angles(sample, rock, match):
    with pytest.raises(ValueError, match=match):
        rocking_scan_geometry(
            SHAPE,
            9.0,
            1.0,
            sample,
            [30.0, 0.0],
            rock,
            0.01,
            diffractometer="34idc",
            detector="34idcTIM2",
        )
