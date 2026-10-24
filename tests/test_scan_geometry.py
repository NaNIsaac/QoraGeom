import numpy as np
import pytest

from qorageom import (
    Detector,
    direct_basis,
    energy_basis,
    reciprocal_basis,
    rocking_scan_geometry,
    wavelength,
)

SHAPE = (32, 32, 16)
LAM = wavelength(9.0) * 1e-10
ANGLES = {"gamma": np.radians(-1.0), "delta": np.radians(37.0)}
SAMPLE = {"th": -71.35, "chi": 85.26, "phi": 12.0}
ARM = {"delta": 37.0, "gamma": -1.0}


def geometry(rock="phi", step=0.01, *, sample=SAMPLE, arm=ARM, energy=9.0, distance=0.5, **kw):
    kw.setdefault("diffractometer", "34idc")
    kw.setdefault("detector", "34idcTIM2")
    return rocking_scan_geometry(SHAPE, energy, distance, sample, arm, rock, step, **kw)


def test_phi_scan_matches_reciprocal_basis():
    geom = geometry()
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


def test_theta_scan_uses_theta_axis():
    geom = geometry("th", 0.001)
    expected = reciprocal_basis(LAM, np.radians(0.001), 0.5, **ANGLES, pixel_size=55e-6)
    np.testing.assert_allclose(geom.B[:, 2], expected[:, 2], rtol=1e-7, atol=1e-6)


def test_energy_scan_uses_energy_basis():
    geom = geometry("energy", 0.0001)
    expected = energy_basis(9.0, 0.0001, 0.5, **ANGLES, pixel_size=55e-6)
    np.testing.assert_allclose(geom.B[:, 2], expected[:, 2])
    assert geom.axis == "energy"
    assert geom.energy_kev == pytest.approx(9.0)


def test_other_angles_energy_and_distance():
    moved = geometry(arm={"delta": 40.0, "gamma": -1.0}, energy=9.5, distance=1.0)
    assert moved.energy_kev == 9.5
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


def test_binning_scales_pixels_and_step():
    one = geometry()
    binned = geometry(binning=(2, 4, 3))
    np.testing.assert_allclose(binned.B[:, 0], 2 * one.B[:, 0])
    np.testing.assert_allclose(binned.B[:, 1], 4 * one.B[:, 1])
    assert np.linalg.norm(binned.B[:, 2]) == pytest.approx(
        3 * np.linalg.norm(one.B[:, 2]), rel=1e-3
    )


def test_resample_returns_lab_volume():
    volume = np.random.default_rng(0).random(SHAPE)
    assert geometry().resample(volume).shape == SHAPE


def test_rocking_about_chi():
    assert geometry("chi").axis == "chi"


def test_id01_and_p10_geometry():
    id01 = geometry(
        "eta",
        sample={"mu": 0.5, "eta": 20.0, "phi": 3.0},
        arm={"nu": 0.7, "delta": 30.0},
        diffractometer="id01",
    )
    p10 = geometry(
        "om",
        sample={"mu": 0.5, "om": 20.0, "chi": 1.0, "phi": 3.0},
        arm={"gam": 0.7, "del": 30.0},
        diffractometer="P10sixc",
    )
    assert (id01.axis, p10.axis) == ("eta", "om")
    assert np.all(np.isfinite(p10.B)) and np.all(np.isfinite(id01.B))


def test_sample_offsets_replace_the_diffractometer_default():
    shifted = geometry(sample_offsets_deg=(0.0, 80.0, 0.0))
    assert not np.allclose(shifted.B, geometry().B)
    np.testing.assert_array_equal(geometry(sample_offsets_deg=(0.0, 90.0, 0.0)).B, geometry().B)


def test_sequence_and_mapping_angles_agree():
    by_order = geometry(sample=[-71.35, 85.26, 12.0], arm=[37.0, -1.0])
    np.testing.assert_array_equal(by_order.B, geometry().B)


def test_custom_detector():
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
        ([0.0, 90.0, 0.0], "delta", "cannot rock"),
    ],
)
def test_bad_angles_and_rocking_motor(sample, rock, match):
    with pytest.raises(ValueError, match=match):
        geometry(rock, sample=sample, arm=[30.0, 0.0])


def test_unknown_names_raise():
    with pytest.raises(KeyError, match="Unknown diffractometer"):
        geometry(diffractometer="nope")
    with pytest.raises(KeyError, match="Unknown detector"):
        geometry(detector="nope")
