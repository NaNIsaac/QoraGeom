import numpy as np
import pytest

from qorageom import (
    HC_KEV_ANGSTROM,
    detector_pose,
    pixel_to_q_lab,
    q_to_tth,
    rotate,
    rotation_matrix,
    shear_correction,
    tth_map,
    tth_to_d,
    tth_to_q,
    wavelength,
)


def test_wavelength_cu_ka():
    assert wavelength(8.04778) == pytest.approx(1.5406, abs=1e-4)
    assert wavelength(HC_KEV_ANGSTROM) == pytest.approx(1.0)


def test_tth_q_round_trip_and_bragg():
    lam = wavelength(10.0)
    tth = np.array([5.0, 30.0, 90.0, 150.0])
    q = tth_to_q(tth, lam)
    np.testing.assert_allclose(q_to_tth(q, lam), tth)
    np.testing.assert_allclose(tth_to_d(tth, lam), 2 * np.pi / q)


def test_q_to_tth_clips_beyond_backscatter():
    lam = 1.0
    assert q_to_tth(4 * np.pi / lam * 1.5, lam) == pytest.approx(180.0)


def test_tth_map_flat_detector():
    sdd, pix = 0.1, 1e-4
    m = tth_map((5, 7), pix, sdd, xc=3, yc=2)
    assert m[2, 3] == pytest.approx(0.0)
    assert m[2, 6] == pytest.approx(np.rad2deg(np.arctan(3 * pix / sdd)))


def test_tth_map_tilted_arm_center_reads_alpha():
    m = tth_map((5, 5), 1e-4, 0.1, xc=2, yc=2, alpha_deg=40.0)
    assert m[2, 2] == pytest.approx(40.0)
    assert m[0, 2] > 40.0 > m[4, 2]  # image up = larger 2θ


def test_rotation_matrix_matches_rotate():
    rng = np.random.default_rng(0)
    axis, v = rng.normal(size=3), rng.normal(size=3)
    R = rotation_matrix(axis, 37.0)
    np.testing.assert_allclose(R @ v, rotate(v, axis, 37.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_shear_correction_unrotated_detector():
    B = shear_correction(wavelength=1e-10, delta_theta=1e-3, D=1.0, gamma=0.0, delta=0.0, pixel_size=5e-5)
    pref = 5e-5 / 1e-10
    np.testing.assert_allclose(B[:, 0], [pref, 0, 0])
    np.testing.assert_allclose(B[:, 1], [0, pref, 0])
    np.testing.assert_allclose(B[:, 2], 0, atol=1e-6)  # q0 = 0 in the forward beam


def test_shear_correction_rocking_column():
    lam, dth, delta = 2.2e-10, np.radians(0.005), np.radians(69.0)
    B = shear_correction(lam, dth, 0.5, 0.0, delta, 5.5e-5)
    q0 = 2 * np.sin(delta / 2) / lam
    # rocking the crystal by dθ moves q along an arc: |Δq| = 2|q0| sin(dθ/2)
    assert np.linalg.norm(B[:, 2]) == pytest.approx(2 * q0 * np.sin(dth / 2), rel=1e-9)
    # the detector columns follow the 2θ arm
    np.testing.assert_allclose(B[:, 0] / np.linalg.norm(B[:, 0]), [np.cos(delta), 0, -np.sin(delta)], atol=1e-12)


def test_pixel_to_q_lab_center_matches_tth_to_q():
    center, x_dir, y_dir = detector_pose(1.0, [1, 0, 0], [0, 1, 0], up_angle_deg=30, side_angle_deg=0)
    q = pixel_to_q_lab(8.04, [1, 0, 0], center, x_dir, y_dir, pixel_size_m=55e-6, px=0, py=0)
    assert np.linalg.norm(q) == pytest.approx(tth_to_q(30.0, wavelength(8.04)))


def test_detector_pose_rejects_non_perpendicular():
    with pytest.raises(ValueError):
        detector_pose(1.0, [1, 0, 0], [1, 1, 0], 10, 0)
