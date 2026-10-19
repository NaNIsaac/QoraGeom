"""Strain along the scattering vector from the phase of a reconstructed Bragg peak."""
import numpy as np

_OFFSETS = (0.0, np.pi / 2, -np.pi / 2)


def _wrap(phase):
    return np.angle(np.exp(1j * phase))


def displacement_gradient(phase, voxel_size, axis):
    """Gradient of ``phase`` [rad] along ``axis``, in rad per length unit of ``voxel_size``.

    The phase is wrapped to (-pi, pi], so a central difference across a 2π jump is wrong.
    The gradient is taken three times, with offsets 0 and ±π/2 added to the phase before
    wrapping, and the smallest magnitude per voxel is kept: the jump sits at a different phase
    value in each copy, so at most one of them is affected. This holds while the true phase
    changes by less than about π/2 over the two voxels of a central difference. Edge voxels use
    one-sided differences, as ``numpy.gradient`` does.

    For the phase of a Bragg peak this is ``G . du/dx`` with ``G`` the scattering vector in
    rad per length; divide by ``|G|`` for the displacement gradient along ``G``.
    """
    phase = np.asarray(phase, dtype=float)
    grads = np.stack(
        [np.gradient(_wrap(phase + offset), voxel_size, axis=axis) for offset in _OFFSETS]
    )
    smallest = np.argmin(np.abs(grads), axis=0)
    return np.take_along_axis(grads, smallest[None], axis=0)[0]


def strain(phase, q_vector, voxel_size, support=None):
    """Strain along the scattering vector, ``d u_G / d r`` along ``G``-hat (dimensionless).

    ``phase`` is the 3D phase of the reconstructed object in radians, taken as ``G . u`` so
    that ``u_G = phase / |G|``; flip its sign if the reconstruction uses the opposite Fourier
    convention. Positive strain is lattice expansion. ``q_vector`` is G in rad per length,
    which includes the 2π that the 2π-free bases of this package leave out, and is in the
    same lab axes as the array axes (axis n of ``phase`` is component n of ``q_vector``);
    ``voxel_size`` is a length in the inverse unit, one value or one per axis.

    Voxels outside the boolean ``support`` are nan. The differences also read the neighbors of
    a voxel, so voxels on the support boundary are only as good as the phase just outside it.
    """
    phase = np.asarray(phase, dtype=float)
    G = np.asarray(q_vector, dtype=float)
    if phase.ndim != G.size:
        raise ValueError(f"phase has {phase.ndim} dimensions but q_vector has {G.size} components")
    sizes = np.broadcast_to(np.asarray(voxel_size, dtype=float), G.shape)
    norm = np.linalg.norm(G)
    along = sum(
        g / norm * displacement_gradient(phase, h, axis)
        for axis, (g, h) in enumerate(zip(G, sizes, strict=True))
        if g != 0
    )
    result = along / norm
    if support is not None:
        result = np.where(np.asarray(support, dtype=bool), result, np.nan)
    return result
