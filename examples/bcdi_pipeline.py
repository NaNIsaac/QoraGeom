"""Simulated BCDI: a strained crystal to its diffraction pattern, back to a strain map.

Needs the examples group: uv run --group examples python examples/bcdi_pipeline.py
"""

import time

import numpy as np
import qoraphase as qp
from scipy.ndimage import binary_erosion

from qorageom import strain

N, M = 32, 16  # array and crystal size in voxels
VOXEL = 5.0  # nm
D_SPACING = 0.2355  # nm, Au 111
G = np.array([1.0, 1.0, 1.0]) / np.sqrt(3) * 2 * np.pi / D_SPACING  # rad/nm
SEED = 0


def crystal():
    """Asymmetric support (so the twin differs) with a localized lattice displacement."""
    r = np.indices((M, M, M)) - (M - 1) / 2
    support = (r[0] + 0.6 * r[1] + 0.3 * r[2] < 3) & ((r**2).sum(0) < (M / 2) ** 2)
    g_hat = G / np.linalg.norm(G)
    s = np.tensordot(g_hat, r, axes=1) * VOXEL  # position along G, nm
    width = 3 * VOXEL
    u = 0.12 * np.exp(-(r**2).sum(0) * VOXEL**2 / (2 * width**2))  # displacement along G, nm
    obj = np.zeros((N, N, N), complex)
    lo = N // 2 - M // 2
    box = (slice(lo, lo + M),) * 3
    obj[box] = support * np.exp(1j * np.linalg.norm(G) * u)
    true_strain = np.full((N, N, N), np.nan)
    # du/ds along G for a spherical Gaussian is -u s / width**2
    true_strain[box] = np.where(support, -u * s / width**2, np.nan)
    return obj, true_strain


def main():
    obj, true_strain = crystal()
    intensity = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2

    start = time.perf_counter()
    best = qp.reconstruct(intensity, starts=4, seed=SEED).best
    elapsed = time.perf_counter() - start

    # translation, global phase and twin branch from the amplitude of the known object
    rec = qp.align_to(best.object, np.abs(obj))
    rec = np.asarray(qp.remove_ramp(rec, np.asarray(best.support), zero_phase=True))

    support = binary_erosion(np.asarray(best.support) & np.isfinite(true_strain))
    estimate = strain(np.angle(rec), G, VOXEL, support)

    error = (estimate - true_strain)[support]
    # the same estimator on the true phase: what voxel size and finite differences cost alone
    floor = (strain(np.angle(obj), G, VOXEL, support) - true_strain)[support]
    print(f"reconstruction: {elapsed:.0f} s, Fourier error {best.error:.1e}")
    print(
        f"true strain:  rms {np.sqrt(np.mean(true_strain[support] ** 2)):.2e}, "
        f"range [{true_strain[support].min():.2e}, {true_strain[support].max():.2e}]"
    )
    print(f"estimator floor on the true phase: rms {np.sqrt(np.mean(floor**2)):.2e}")
    print(f"strain error: rms {np.sqrt(np.mean(error**2)):.2e}, max {np.abs(error).max():.2e}")


if __name__ == "__main__":
    main()
