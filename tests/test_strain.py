import numpy as np
import pytest

from qorageom import displacement_gradient, strain

SHAPE = (40, 40, 40)
VOXEL = 0.8  # nm
GHAT = np.array([1.0, 2.0, 2.0]) / 3
G = 26.0 * GHAT  # rad/nm, an fcc {111}-sized vector
INNER = (slice(1, -1),) * 3  # edge voxels use one-sided differences


def _along_g():
    """Lab position of every voxel projected on G-hat [nm]."""
    r = np.indices(SHAPE).astype(float) * VOXEL
    return np.tensordot(GHAT, r, axes=1)


def test_linear_displacement_gives_constant_strain_through_many_wraps():
    eps = 0.03
    phase = np.linalg.norm(G) * eps * _along_g()  # u_G = eps * (r . G-hat)
    assert phase.max() > 8 * np.pi  # wrapped several times
    wrapped = np.angle(np.exp(1j * phase))
    result = strain(wrapped, G, VOXEL)
    np.testing.assert_allclose(result[INNER], eps, atol=1e-9)


def test_quadratic_displacement_gives_linear_strain():
    a = 0.001  # strain = a * s, s the position along G-hat [nm]
    s = _along_g()
    phase = np.linalg.norm(G) * a * s**2 / 2
    assert phase.max() > 6 * np.pi
    result = strain(np.angle(np.exp(1j * phase)), G, VOXEL)
    np.testing.assert_allclose(result[INNER], (a * s)[INNER], atol=1e-9)


def test_wrapped_phase_matches_unwrapped_phase():
    phase = 40.0 * np.sin(_along_g() / 6.0)  # steps stay below pi/2 per two voxels
    result = strain(phase, G, VOXEL)
    wrapped = strain(np.angle(np.exp(1j * phase)), G, VOXEL)
    np.testing.assert_allclose(wrapped, result, atol=1e-9)


def test_plain_gradient_of_wrapped_phase_fails_where_robust_one_works():
    phase = 2.5 * _along_g()  # 0.67 rad per voxel along axis 0
    wrapped = np.angle(np.exp(1j * phase))
    naive = np.gradient(wrapped, VOXEL, axis=0)
    robust = displacement_gradient(wrapped, VOXEL, axis=0)
    expected = np.gradient(phase, VOXEL, axis=0)
    assert np.abs(naive - expected)[INNER].max() > 1.0
    np.testing.assert_allclose(robust[INNER], expected[INNER], atol=1e-9)


def test_axis_aligned_vector_and_per_axis_voxel_size():
    x = np.indices((20, 20, 20))[0] * 0.5
    phase = 10.0 * 0.02 * x  # G along x, strain 0.02
    result = strain(phase, [10.0, 0, 0], [0.5, 0.7, 0.9])
    np.testing.assert_allclose(result[INNER[0], 2:-2, 2:-2][1:-1], 0.02, atol=1e-9)


def test_support_outside_is_nan():
    phase = np.zeros((6, 6, 6))
    support = np.zeros(phase.shape, bool)
    support[2:4, 2:4, 2:4] = True
    result = strain(phase, G, VOXEL, support)
    assert np.isnan(result[~support]).all()
    np.testing.assert_array_equal(result[support], 0.0)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        strain(np.zeros((4, 4)), G, VOXEL)
