import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from qorageom import (
    DIFFRACTOMETERS,
    Diffractometer,
    get_diffractometer,
    reciprocal_basis,
    rotation_matrix,
)
from qorageom.diffractometer import axis_vector


@pytest.mark.parametrize(
    "axis, vector",
    [
        ("x+", [1, 0, 0]),
        ("y-", [0, -1, 0]),
        ("z+", [0, 0, 1]),
    ],
)
def test_axis_vector(axis, vector):
    np.testing.assert_array_equal(axis_vector(axis), vector)


@pytest.mark.parametrize("axis", ["", "y", "w+", "y*", "y++"])
def test_axis_vector_rejects_malformed(axis):
    with pytest.raises(ValueError):
        axis_vector(axis)


def test_34idc_sample_rotation_is_theta_chi_phi_from_outer_to_inner():
    th, chi, phi = 12.0, -30.0, 47.0
    Ry, Rz = (lambda a: rotation_matrix([0, 1, 0], a)), (lambda a: rotation_matrix([0, 0, 1], a))
    expected = Ry(th) @ Rz(-chi) @ Ry(phi)
    np.testing.assert_allclose(
        get_diffractometer("34idc").sample_rotation([th, chi, phi]), expected
    )


def test_34idc_detector_rotation_matches_reciprocal_basis_convention():
    # the detector direction at delta, gamma is (sin d cos g, sin g, cos d cos g)
    delta, gamma = 37.0, -1.0
    d = get_diffractometer("34idc").detector_rotation([delta, gamma]) @ [0, 0, 1]
    dr, gr = np.radians([delta, gamma])
    np.testing.assert_allclose(d, [np.sin(dr) * np.cos(gr), np.sin(gr), np.cos(dr) * np.cos(gr)])


@pytest.mark.parametrize("name", sorted(DIFFRACTOMETERS))
def test_rotations_are_proper_and_identity_at_zero(name):
    diff = get_diffractometer(name)
    for R in (
        diff.sample_rotation(np.full(len(diff.sample_axes), 0.0)),
        diff.detector_rotation(np.full(len(diff.detector_axes), 0.0)),
    ):
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)
    rng = np.random.default_rng(3)
    R = diff.sample_rotation(rng.uniform(-90, 90, len(diff.sample_axes)))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_registry_has_the_cohere_beamlines():
    assert set(DIFFRACTOMETERS) == {"34idc", "P10sixc", "id01"}
    assert get_diffractometer("id01").sample_motors == ("mu", "eta", "phi")


def test_unknown_diffractometer_and_bad_angle_count():
    with pytest.raises(KeyError):
        get_diffractometer("nope")
    with pytest.raises(ValueError):
        get_diffractometer("34idc").sample_rotation([1.0, 2.0])


def test_dataclass_validates_lengths_and_axes():
    with pytest.raises(ValueError):
        Diffractometer("bad", ("y+",), ("a", "b"), (), ())
    with pytest.raises(ValueError):
        Diffractometer("bad", ("q+",), ("a",), (), ())


# independent construction: scipy rotations built from the axis strings, central-pixel q at
# two angles as cohere/xrayutilities does it
LAM, D, PIX = 1.2e-10, 0.5, 55e-6


def _rot(axis, angle_deg):
    return Rotation.from_rotvec(np.radians(angle_deg) * axis_vector(axis)).as_matrix()


def _stack_rot(axes, angles_deg):
    R = np.eye(3)
    for axis, angle in zip(axes, angles_deg, strict=True):
        R = R @ _rot(axis, angle)
    return R


def _q_sample(diff, sample, detector, wavelength=LAM, pixel=(0.0, 0.0), orientation=("x+", "y+")):
    """q in the sample frame, no 2π, of the pixel (i, j) away from the center."""
    beam = np.array(diff.incident_axis)
    pos = D * beam + PIX * (
        pixel[0] * axis_vector(orientation[0]) + pixel[1] * axis_vector(orientation[1])
    )
    kf = _stack_rot(diff.detector_axes, detector) @ pos / np.linalg.norm(pos)
    return _stack_rot(diff.sample_axes, sample).T @ (kf - beam) / wavelength


CASES = [
    ("P10sixc", [0.4, 21.0, 3.0, -35.0], [0.7, 31.0], "om"),
    ("P10sixc", [0.4, 21.0, 3.0, -35.0], [0.7, 31.0], "phi"),
    ("id01", [0.5, 18.0, -35.0], [0.7, 30.0], "eta"),
    ("id01", [0.5, 18.0, -35.0], [0.7, 30.0], "mu"),
    ("34idc", [-71.35, 85.26, 12.0], [37.0, -1.0], "chi"),
]


@pytest.mark.parametrize("name, sample, detector, rock", CASES)
def test_rocking_basis_k_matches_q_at_two_angles(name, sample, detector, rock):
    diff = get_diffractometer(name)
    step = 0.02
    B = diff.rocking_basis(LAM, D, PIX, sample, detector, rock, step)

    offsets = np.array(diff.sample_offsets_deg or np.zeros(len(sample)))
    start = np.array(sample) - offsets
    moved = start.copy()
    moved[diff.sample_motors.index(rock)] += step
    dq_sample = _q_sample(diff, moved, detector) - _q_sample(diff, start, detector)
    expected = _stack_rot(diff.sample_axes, start) @ dq_sample  # back to the starting lab frame
    np.testing.assert_allclose(B[:, 2], expected, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("name, sample, detector, rock", CASES)
def test_rocking_basis_pixel_columns_match_a_pixel_step(name, sample, detector, rock):
    diff = get_diffractometer(name)
    orientation = ("x+", "y-")
    B = diff.rocking_basis(LAM, D, PIX, sample, detector, rock, 0.02, orientation)
    start = np.array(sample) - np.array(diff.sample_offsets_deg or np.zeros(len(sample)))
    R = _stack_rot(diff.sample_axes, start)
    base = _q_sample(diff, start, detector, orientation=orientation)
    for column, pixel in ((0, (1, 0)), (1, (0, 1))):
        step = R @ (_q_sample(diff, start, detector, pixel=pixel, orientation=orientation) - base)
        np.testing.assert_allclose(
            B[:, column], step, rtol=1e-3, atol=1e-3 * np.linalg.norm(B[:, column])
        )


@pytest.mark.parametrize("rock, axis", [("th", "theta"), ("phi", "phi")])
def test_rocking_basis_reproduces_reciprocal_basis_for_34idc(rock, axis):
    gamma, delta = np.radians([-1.0, 37.0])
    sample = [-71.35, 85.26, 12.0]
    generic = get_diffractometer("34idc").rocking_basis(
        LAM, D, PIX, sample, [37.0, -1.0], rock, 0.01
    )
    ref = reciprocal_basis(
        LAM,
        np.radians(0.01),
        D,
        gamma,
        delta,
        PIX,
        theta=np.radians(sample[0]),
        chi=np.radians(sample[1]),
        axis=axis,
    )
    np.testing.assert_allclose(generic, ref, rtol=1e-12, atol=1e-6)


def test_rocking_basis_energy_and_validation():
    diff = get_diffractometer("34idc")
    B = diff.rocking_basis(LAM, D, PIX, [0, 90, 0], [37.0, -1.0], "energy", 1e-4)
    q0 = (diff.detector_rotation([37.0, -1.0]) @ [0, 0, 1] - [0, 0, 1]) / LAM
    np.testing.assert_allclose(B[:, 2], 1e-4 * q0)
    with pytest.raises(ValueError, match="cannot rock"):
        diff.rocking_basis(LAM, D, PIX, [0, 90, 0], [37.0, -1.0], "delta", 0.1)
    with pytest.raises(ValueError, match="perpendicular"):
        diff.rocking_basis(LAM, D, PIX, [0, 90, 0], [37.0, -1.0], "phi", 0.1, ("z+", "y+"))
