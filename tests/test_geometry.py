import numpy as np
import pytest

from qorageom import (
    HC_KEV_ANGSTROM,
    basis_diagnostics,
    detector_pose,
    direct_basis,
    energy_basis,
    energy_to_kev,
    lab_voxel_size,
    pixel_to_q_lab,
    q_to_tth,
    reciprocal_basis,
    resample_to_lab,
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
    B = shear_correction(
        wavelength=1e-10, delta_theta=1e-3, D=1.0, gamma=0.0, delta=0.0, pixel_size=5e-5
    )
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
    np.testing.assert_allclose(
        B[:, 0] / np.linalg.norm(B[:, 0]), [np.cos(delta), 0, -np.sin(delta)], atol=1e-12
    )


def test_pixel_to_q_lab_center_matches_tth_to_q():
    center, x_dir, y_dir = detector_pose(
        1.0, [1, 0, 0], [0, 1, 0], up_angle_deg=30, side_angle_deg=0
    )
    q = pixel_to_q_lab(8.04, [1, 0, 0], center, x_dir, y_dir, pixel_size_m=55e-6, px=0, py=0)
    assert np.linalg.norm(q) == pytest.approx(tth_to_q(30.0, wavelength(8.04)))


def test_detector_pose_rejects_non_perpendicular():
    with pytest.raises(ValueError):
        detector_pose(1.0, [1, 0, 0], [1, 1, 0], 10, 0)


def test_pixel_to_q_lab_array_matches_per_pixel():
    center, x_dir, y_dir = detector_pose(
        1.0, [1, 0, 0], [0, 1, 0], up_angle_deg=30, side_angle_deg=10
    )
    args = (8.04, [1, 0, 0], center, x_dir, y_dir, 55e-6)
    py, px = np.mgrid[-2:3, -3:4]
    q = pixel_to_q_lab(*args, px, py)
    assert q.shape == (5, 7, 3)
    for i, j in np.ndindex(px.shape):
        np.testing.assert_array_equal(q[i, j], pixel_to_q_lab(*args, px[i, j], py[i, j]))


def _sample_stack(theta, chi, phi, chi_offset):
    """Sample-to-lab rotation, outer to inner theta (y+), chi (z-), phi (y+), angles in degrees."""
    ry = lambda a: rotation_matrix([0, 1, 0], a)
    return ry(theta) @ rotation_matrix([0, 0, -1], chi - chi_offset) @ ry(phi)


ANGLES = {
    "lam": 1.2e-10,
    "D": 1.0,
    "gamma": np.radians(-1.0),
    "delta": np.radians(37.0),
    "pix": 5.5e-5,
}


@pytest.mark.parametrize(
    "axis, theta0, chi0, phi0",
    [
        ("theta", -71.3, 85.3, 12.0),
        ("theta", 20.0, 90.0, 0.0),
        ("phi", -71.3, 85.3, 12.0),
        ("phi", 15.0, 60.0, -30.0),
    ],
)
def test_reciprocal_basis_k_matches_explicit_sample_rotation(axis, theta0, chi0, phi0):
    a = ANGLES
    step = 0.01
    B = reciprocal_basis(
        a["lam"],
        np.radians(step),
        a["D"],
        a["gamma"],
        a["delta"],
        a["pix"],
        theta=np.radians(theta0),
        chi=np.radians(chi0),
        axis=axis,
    )

    # fixed detector sees the lab q Q; the sample point it samples after the step is
    # R(motor + step)^-1 Q, expressed back in the starting orientation
    s1, s2, s3 = np.eye(3)
    B_det = rotation_matrix(s2, np.rad2deg(a["delta"])) @ rotation_matrix(
        -s1, np.rad2deg(a["gamma"])
    )
    Q = (B_det @ s3 - s3) / a["lam"]
    start = (theta0, chi0, phi0)
    moved = (theta0 + step, chi0, phi0) if axis == "theta" else (theta0, chi0, phi0 + step)
    R0 = _sample_stack(*start, 90.0)
    R1 = _sample_stack(*moved, 90.0)
    np.testing.assert_allclose(B[:, 2], R0 @ R1.T @ Q - Q, rtol=1e-8, atol=1e-3)


def test_reciprocal_basis_theta_agrees_with_shear_correction_to_first_order():
    a, step = ANGLES, np.radians(0.005)
    new = reciprocal_basis(a["lam"], step, a["D"], a["gamma"], a["delta"], a["pix"])
    old = shear_correction(a["lam"], step, a["D"], a["gamma"], a["delta"], a["pix"])
    np.testing.assert_allclose(new[:, :2], old[:, :2])
    np.testing.assert_allclose(new[:, 2], old[:, 2], rtol=1e-3)
    assert not np.allclose(new[:, 2], old[:, 2], rtol=1e-12)  # second-order term differs


def test_reciprocal_basis_phi_with_chi_offset_removed_is_theta_rocking():
    a = ANGLES
    step = np.radians(0.01)
    # chi - chi_offset = 0 and theta = 0 leave the phi axis along lab y
    phi = reciprocal_basis(
        a["lam"],
        step,
        a["D"],
        a["gamma"],
        a["delta"],
        a["pix"],
        theta=0.0,
        chi=np.pi / 2,
        axis="phi",
    )
    theta = reciprocal_basis(a["lam"], step, a["D"], a["gamma"], a["delta"], a["pix"])
    np.testing.assert_allclose(phi, theta, atol=1e-6)


def test_reciprocal_basis_rejects_unknown_axis():
    with pytest.raises(ValueError):
        reciprocal_basis(1e-10, 1e-4, 1.0, 0.0, 0.5, 5e-5, axis="chi")


def test_direct_basis_is_dual_of_reciprocal_basis():
    a = ANGLES
    B = reciprocal_basis(a["lam"], np.radians(0.01), a["D"], a["gamma"], a["delta"], a["pix"])
    A = direct_basis(B)
    np.testing.assert_allclose(B.T @ A, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(A, np.linalg.inv(B).T, rtol=1e-9)


def test_direct_basis_shape_gives_fft_voxel_vectors():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(3, 3))
    shape = (4, 6, 8)
    A = direct_basis(B, shape)
    np.testing.assert_allclose(A * shape, direct_basis(B))
    # one voxel step along axis i, times the FFT length, is a full reciprocal period
    for i in range(3):
        np.testing.assert_allclose(B.T @ (A[:, i] * shape[i]), np.eye(3)[i], atol=1e-12)


def test_direct_basis_rejects_singular_basis():
    B = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    with pytest.raises(ValueError):
        direct_basis(B)
    with pytest.raises(ValueError):
        direct_basis(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]))


def test_basis_diagnostics_orthogonal_and_sheared():
    d = basis_diagnostics(np.diag([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(d["lengths"], [1, 2, 4])
    np.testing.assert_allclose(d["angles_deg"], [90, 90, 90])
    assert d["cond"] == pytest.approx(4.0)

    sheared = np.array([[1.0, 0, 1.0], [0, 1.0, 0], [0, 0, 1.0]])  # third column 45 deg from first
    d = basis_diagnostics(sheared)
    assert d["angles_deg"][1] == pytest.approx(45.0)
    assert d["cond"] > 1.0


def test_lab_voxel_size_takes_smallest_edge():
    lam, D, pix = 1.2e-10, 1.0, 5.5e-5
    edge_xy = lam * D / pix / 64
    # rocking edge lam / (step N3) is 1.2x the detector edge, inside the 0.7 floor
    step = lam / (1.2 * edge_xy * 32)
    assert lab_voxel_size(lam, D, pix, (64, 64, 32), step) == pytest.approx(edge_xy)


def test_lab_voxel_size_fov_floor():
    lam, D, pix = 1.2e-10, 1.0, 5.5e-5
    edge_xy = lam * D / pix / 64
    step = lam / (10 * edge_xy * 32)  # rocking edge 10x larger
    assert lab_voxel_size(lam, D, pix, (64, 64, 32), step) == pytest.approx(0.7 * 10 * edge_xy)
    assert lab_voxel_size(lam, D, pix, (64, 64, 32), step, fov_min=0) == pytest.approx(edge_xy)


def test_resample_to_lab_identity_basis_is_a_no_op():
    rng = np.random.default_rng(2)
    vol = rng.normal(size=(6, 7, 5))
    voxel = 2e-9
    B = np.diag(1 / (np.array(vol.shape) * voxel))
    np.testing.assert_allclose(resample_to_lab(vol, B, voxel), vol, atol=1e-12)


def test_resample_to_lab_unshears_a_smooth_object():
    shape, voxel = (33, 33, 33), 1.0
    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0]])  # detector voxel vectors
    A = shear * 1.0
    B = np.linalg.inv(A * np.array(shape)).T  # so that direct_basis(B, shape) == A
    np.testing.assert_allclose(direct_basis(B, shape), A, atol=1e-12)

    center = np.array(shape) // 2

    def gauss(r):
        return np.exp(-np.sum((r - np.array([[1.0], [-2.0], [0.5]])) ** 2, axis=0) / (2 * 4.0**2))

    idx = np.indices(shape).reshape(3, -1) - center[:, None]
    det_volume = gauss(A @ idx).reshape(shape)
    lab = resample_to_lab(det_volume, B, voxel, order=3)
    expected = gauss(idx * voxel).reshape(shape)
    inner = (slice(8, -8),) * 3  # away from the zero-filled corners
    np.testing.assert_allclose(lab[inner], expected[inner], atol=5e-3)


def test_resample_to_lab_keeps_complex_phase():
    shape, voxel = (9, 9, 9), 1.0
    B = np.diag(1 / (np.array(shape) * voxel))
    vol = np.exp(1j * np.linspace(0, 1, 9**3)).reshape(shape)
    out = resample_to_lab(vol, B, voxel)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out, vol, atol=1e-12)


@pytest.mark.parametrize("shape", [(8, 8, 8), (8, 10, 6), (9, 8, 7)])
def test_resample_to_lab_keeps_the_origin_voxel_fixed(shape):
    rng = np.random.default_rng(4)
    B = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    vol = np.zeros(shape)
    origin = tuple(n // 2 for n in shape)
    vol[origin] = 1.0
    out = resample_to_lab(vol, B, voxel_size=0.5)
    assert np.unravel_index(np.argmax(out), shape) == origin
    assert out[origin] == pytest.approx(1.0)


def test_energy_to_kev_detects_unit():
    assert energy_to_kev(9.0, 0.002) == (9.0, 0.002)
    kev, dkev = energy_to_kev(9000.0, 2.0)
    assert kev == pytest.approx(9.0)
    assert dkev == pytest.approx(0.002)


def test_energy_basis_k_matches_pixel_q_at_two_energies():
    # pixel_to_q_lab builds q from the two wavevectors, independent of the basis algebra
    delta_deg, energy, dE, D = 37.0, 9.0, 0.003, 0.5
    beam = [0, 0, 1]
    center, x_dir, y_dir = detector_pose(
        D, beam, [0, 1, 0], up_angle_deg=0, side_angle_deg=delta_deg
    )

    def q_lab(e):  # 1/Å with 2π -> 1/m without
        return pixel_to_q_lab(e, beam, center, x_dir, y_dir, 55e-6, 0, 0) / (2 * np.pi) * 1e10

    B = energy_basis(energy, dE, D, 0.0, np.radians(delta_deg), 55e-6)
    np.testing.assert_allclose(B[:, 2], q_lab(energy + dE) - q_lab(energy), rtol=1e-6)


def test_energy_basis_ev_equals_kev_and_pixel_columns_match_shear_correction():
    args = (0.5, np.radians(-1.0), np.radians(37.0), 55e-6)
    kev = energy_basis(9.0, 0.003, *args)
    np.testing.assert_allclose(energy_basis(9000.0, 3.0, *args), kev)
    ref = shear_correction(wavelength(9.0) * 1e-10, 1e-4, *args)
    np.testing.assert_allclose(kev[:, :2], ref[:, :2])
