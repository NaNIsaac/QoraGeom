import numpy as np
from math import tau

HC_KEV_ANGSTROM = 12.398419843320026  # CODATA 2018 hc [keV·Å]


def wavelength(energy_kev):
    """X-ray wavelength [Å] at photon energy [keV]."""
    return HC_KEV_ANGSTROM / np.asarray(energy_kev, dtype=float)


def tth_to_q(tth_deg, wavelength_a):
    """|q| [1/Å] = 4π sin(θ) / λ."""
    return 4 * np.pi * np.sin(np.deg2rad(np.asarray(tth_deg, dtype=float) / 2)) / wavelength_a


def q_to_tth(q, wavelength_a):
    """2θ [deg] of |q| [1/Å]; |q| beyond 4π/λ is clipped to 180°."""
    s = np.clip(np.asarray(q, dtype=float) * wavelength_a / (4 * np.pi), -1, 1)
    return 2 * np.rad2deg(np.arcsin(s))


def tth_to_d(tth_deg, wavelength_a):
    """Bragg d-spacing [Å] at 2θ [deg]."""
    return wavelength_a / (2 * np.sin(np.deg2rad(np.asarray(tth_deg, dtype=float) / 2)))


def tth_map(shape, pixel_m, sdd_m, xc, yc, alpha_deg=0.0):
    """True 2θ [deg] of every pixel of a flat detector.

    The detector arm is tilted by ``alpha_deg`` in the vertical plane; (xc, yc) is the
    PONI in pixels and image rows grow downward, so up on the image is +2θ.
      P = SDD·n + u·ex + v·ev,  n = (0, sinα, cosα),  ev = (0, cosα, -sinα)
      cos 2θ = (SDD·cosα - v·sinα) / sqrt(SDD² + u² + v²)
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    u = (x - xc) * pixel_m
    v = (yc - y) * pixel_m
    a = np.deg2rad(alpha_deg)
    cos_tth = (sdd_m * np.cos(a) - v * np.sin(a)) / np.sqrt(sdd_m**2 + u**2 + v**2)
    return np.rad2deg(np.arccos(np.clip(cos_tth, -1, 1)))


def rotate(vector, axis, angle_deg):
    v = np.asarray(vector, dtype=float)
    a = np.array(axis, dtype=float, copy=True)
    a /= np.linalg.norm(a)

    angle = np.deg2rad(angle_deg)
    return (
        v * np.cos(angle)
        + np.cross(a, v) * np.sin(angle)
        + a * np.dot(a, v) * (1 - np.cos(angle))
    )


def rotation_matrix(axis, angle_deg):
    """Matrix form of ``rotate``: ``rotation_matrix(a, t) @ v == rotate(v, a, t)``."""
    return np.column_stack([rotate(e, axis, angle_deg) for e in np.eye(3)])


def shear_correction(wavelength, delta_theta, D, gamma, delta, pixel_size):
    """Detector-frame to reciprocal-space basis B_recip for a BCDI rocking scan (34-ID-C).

    Columns are q per detector pixel (i, j) and per rocking step (k), so
    ``q = B_recip @ (i, j, k)``. ``wavelength``, ``D`` and ``pixel_size`` share one
    length unit, angles are in radians, and q comes out in inverse length without 2π.
    gamma turns about -x because the 34-ID-C arm rotates clockwise.
    """
    s1, s2, s3 = np.eye(3)
    B_det = rotation_matrix(s2, np.rad2deg(delta)) @ rotation_matrix(-s1, np.rad2deg(gamma))

    pref = pixel_size / (wavelength * D)
    qi = pref * B_det[:, 0]
    qj = pref * B_det[:, 1]

    q0 = (B_det @ s3 - s3) / wavelength
    qk = q0 - rotation_matrix(s2, np.rad2deg(delta_theta)) @ q0
    return np.stack([qi, qj, qk], axis=1)


def reciprocal_basis(
    wavelength, delta_rock, D, gamma, delta, pixel_size, theta=0.0, chi=0.0, *,
    axis="theta", chi_offset=np.pi / 2,
):
    """Detector-frame to reciprocal-space basis for a rocking scan about theta or phi (34-ID-C).

    Same layout and units as ``shear_correction``: ``q = B @ (i, j, k)`` in inverse length
    without 2π, lengths in one unit, angles in radians. The sample circles are
    theta (y+), chi (z-) and phi (y+) from outer to inner, and the chi motor reads
    ``chi_offset`` more than the physical angle, so chi - chi_offset is what turns the sample.

    ``axis="theta"`` rocks about lab y. ``axis="phi"`` rocks about y carried through theta and chi.
    The k column is the exact q step ``R(-delta_rock) Q - Q`` of the rocking rotation, whereas
    ``shear_correction`` uses ``Q - R(delta_rock) Q``; the two agree to first order in the step.
    """
    if axis not in ("theta", "phi"):
        raise ValueError(f"axis must be 'theta' or 'phi', got {axis!r}")

    s1, s2, s3 = np.eye(3)
    B_det = rotation_matrix(s2, np.rad2deg(delta)) @ rotation_matrix(-s1, np.rad2deg(gamma))

    pref = pixel_size / (wavelength * D)
    qi = pref * B_det[:, 0]
    qj = pref * B_det[:, 1]

    q0 = (B_det @ s3 - s3) / wavelength
    outer = np.eye(3)
    if axis == "phi":
        outer = rotation_matrix(s2, np.rad2deg(theta)) @ rotation_matrix(
            -s3, np.rad2deg(chi - chi_offset)
        )
    qk = outer @ rotation_matrix(s2, -np.rad2deg(delta_rock)) @ outer.T @ q0 - q0
    return np.stack([qi, qj, qk], axis=1)


def direct_basis(B, shape=None):
    """Real-space basis conjugate to the reciprocal basis ``B`` (columns are q per step, no 2π).

    Columns ``a_i`` satisfy ``a_i . q_j = delta_ij``, so ``B.T @ direct_basis(B) == I``. With
    ``shape = (N1, N2, N3)`` each column is divided by its array length, giving the voxel
    vectors of the FFT grid, in the length unit of 1/q. The legacy MATLAB basis carries a 2π
    in both bases and an overall sign from a reversed triple product; neither is reproduced.
    """
    B = np.asarray(B, dtype=float)
    b1, b2, b3 = B.T
    volume = np.dot(b1, np.cross(b2, b3))
    if abs(volume) <= np.finfo(float).eps * np.prod(np.linalg.norm(B, axis=0)):
        raise ValueError("reciprocal basis is singular")
    A = np.stack([np.cross(b2, b3), np.cross(b3, b1), np.cross(b1, b2)], axis=1) / volume
    if shape is not None:
        A = A / np.asarray(shape, dtype=float)
    return A


def basis_diagnostics(B):
    """Column lengths, pairwise column angles [deg] and condition number of a 3x3 basis.

    Angles are ordered (1-2, 1-3, 2-3). A condition number above about 10 means the
    angles of the scan leave the voxel grid badly sheared.
    """
    B = np.asarray(B, dtype=float)
    lengths = np.linalg.norm(B, axis=0)
    unit = B / lengths
    pairs = [(0, 1), (0, 2), (1, 2)]
    cos = np.clip([unit[:, i] @ unit[:, j] for i, j in pairs], -1, 1)
    return {
        "lengths": lengths,
        "angles_deg": np.rad2deg(np.arccos(cos)),
        "cond": np.linalg.cond(B),
    }


def pixel_to_q_lab(
    energy_kev,
    incident_dir,
    center_pos,
    pixel_x_dir,
    pixel_y_dir,
    pixel_size_m,
    px,
    py,
):
    """Lab-frame q [1/Å] of pixel (px, py), counted from the detector center.

    ``px`` and ``py`` may be arrays that broadcast together; q is then stacked along a
    trailing axis of length 3.
    """
    incident = np.asarray(incident_dir, dtype=float)
    incident = incident / np.linalg.norm(incident)

    px = np.asarray(px, dtype=float)[..., None]
    py = np.asarray(py, dtype=float)[..., None]
    pixel_pos = (
        np.asarray(center_pos, dtype=float)
        + pixel_size_m * px * np.asarray(pixel_x_dir, dtype=float)
        + pixel_size_m * py * np.asarray(pixel_y_dir, dtype=float)
    )

    k = tau * energy_kev / HC_KEV_ANGSTROM  # [Å^-1]
    ki = k * incident
    kf = k * pixel_pos / np.linalg.norm(pixel_pos, axis=-1, keepdims=True)
    return kf - ki


def detector_pose(
    distance_m,
    incident_dir,
    scatter_up,
    up_angle_deg,
    side_angle_deg,
):
    """예제 장비의 각도에서 검출기 중심과 두 픽셀 방향을 구한다."""
    incident = np.asarray(incident_dir, dtype=float)
    up = np.asarray(scatter_up, dtype=float)
    incident = incident / np.linalg.norm(incident)
    up = up / np.linalg.norm(up)

    if not np.isclose(np.dot(incident, up), 0.0):
        raise ValueError("incident_dir and scatter_up must be perpendicular")

    side = np.cross(up, incident)

    def turn(v):
        v = rotate(v, -side, up_angle_deg)
        return rotate(v, up, side_angle_deg)

    center_pos = distance_m * turn(incident)
    pixel_x_dir = turn(side)
    pixel_y_dir = turn(up)
    return center_pos, pixel_x_dir, pixel_y_dir
