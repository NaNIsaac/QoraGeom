"""Reciprocal and direct bases of a rocking-scan volume, from plain values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from qorageom.detector import Detector, get_detector
from qorageom.diffractometer import Diffractometer, get_diffractometer
from qorageom.geometry import (
    basis_diagnostics,
    direct_basis,
    lab_voxel_size,
    resample_to_lab,
    wavelength,
)


@dataclass(frozen=True, eq=False)
class ScanGeometry:
    """Lengths in metres, q in 1/m without 2π, axes ordered (detector axis 0, axis 1, rocking).

    ``B`` is the reciprocal basis (``q = B @ (i, j, k)``), ``A`` the conjugate direct basis of
    the FFT grid and ``voxel_size`` the edge used by ``resample``.
    """

    diffractometer: str
    detector: str
    axis: str
    shape: tuple[int, int, int]
    wavelength_m: float
    energy_kev: float
    distance_m: float
    B: np.ndarray
    A: np.ndarray
    voxel_size: float
    diagnostics: dict

    def resample(self, volume, order=1):
        """Resample a volume of ``shape`` onto the orthogonal lab grid of this scan."""
        return resample_to_lab(volume, self.B, self.voxel_size, order)


def _ordered(angles, motors: tuple[str, ...], what: str) -> list[float]:
    if isinstance(angles, Mapping):
        unknown = set(angles) - set(motors)
        missing = [m for m in motors if m not in angles]
        if unknown or missing:
            raise ValueError(
                f"{what} angles need exactly {list(motors)}; "
                f"missing {missing}, unknown {sorted(unknown)}"
            )
        return [angles[m] for m in motors]
    if len(angles) != len(motors):
        raise ValueError(f"{what} angles need {len(motors)} values in the order {list(motors)}")
    return list(angles)


def rocking_scan_geometry(
    shape: Sequence[int],
    energy_kev: float,
    distance_m: float,
    sample_angles_deg: Sequence[float] | Mapping[str, float],
    detector_angles_deg: Sequence[float] | Mapping[str, float],
    rock: str,
    step: float,
    *,
    diffractometer: str | Diffractometer,
    detector: str | Detector,
    binning: Sequence[int] = (1, 1, 1),
    sample_offsets_deg: Sequence[float] | None = None,
) -> ScanGeometry:
    """Geometry of a rocking scan from the circle readings, without any file format.

    ``sample_angles_deg`` and ``detector_angles_deg`` are motor readings in degrees, either in
    the order of the diffractometer's ``sample_motors`` and ``detector_motors`` or as a mapping
    from those names. ``rock`` is one of the ``sample_motors``, stepped by ``step`` degrees per
    measured frame, or ``"energy"``, stepped by ``step`` keV per frame. ``sample_offsets_deg``
    replaces the diffractometer's offsets (motor reading minus circle angle). ``shape`` is the
    (N1, N2, N3) of the volume; ``binning`` the detector binning of the data and the number of
    frames averaged per rocking step. The basis is ``Diffractometer.rocking_basis`` with the
    detector's orientation.
    """
    diff = diffractometer
    if not isinstance(diff, Diffractometer):
        diff = get_diffractometer(diff)
    if sample_offsets_deg is not None:
        diff = replace(diff, sample_offsets_deg=tuple(sample_offsets_deg))
    det = detector if isinstance(detector, Detector) else get_detector(detector)
    if rock != "energy" and rock not in diff.sample_motors:
        raise ValueError(
            f"cannot rock about {rock!r}; supported: {list(diff.sample_motors)} and energy"
        )

    sample = _ordered(sample_angles_deg, diff.sample_motors, "sample")
    arm = _ordered(detector_angles_deg, diff.detector_motors, "detector")
    step = step * binning[2]
    pixel = (det.pixel_size[0] * binning[0], det.pixel_size[1] * binning[1])
    lam = wavelength(energy_kev) * 1e-10
    B = diff.rocking_basis(
        lam,
        distance_m,
        pixel,
        sample,
        arm,
        rock,
        step / energy_kev if rock == "energy" else step,
        det.orientation,
    )
    # for energy the angular step of lab_voxel_size is the one whose edge is 1 / (|q_k| N3)
    rock_step = lam * np.linalg.norm(B[:, 2]) if rock == "energy" else np.radians(step)

    shape = tuple(int(n) for n in shape)
    return ScanGeometry(
        diffractometer=diff.name,
        detector=det.name,
        axis=rock,
        shape=shape,
        wavelength_m=lam,
        energy_kev=energy_kev,
        distance_m=distance_m,
        B=B,
        A=direct_basis(B, shape),
        voxel_size=lab_voxel_size(lam, distance_m, pixel, shape, rock_step),
        diagnostics=basis_diagnostics(B),
    )


def __getattr__(name):
    # compatibility path for qorageom.scan_geometry.geometry_from_spec; the adapter owns it now
    if name == "geometry_from_spec":
        import warnings

        from qorageom.adapters import spec

        warnings.warn(
            "qorageom.scan_geometry.geometry_from_spec moved to "
            "qorageom.adapters.spec.geometry_from_spec",
            DeprecationWarning,
            stacklevel=2,
        )
        return spec.geometry_from_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
