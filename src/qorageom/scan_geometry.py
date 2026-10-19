"""One call from a SPEC scan to the reciprocal and direct bases of its volume."""
from dataclasses import dataclass

import numpy as np

from qorageom.detector import get_detector
from qorageom.diffractometer import get_diffractometer
from qorageom.geometry import (
    basis_diagnostics,
    direct_basis,
    energy_basis,
    energy_to_kev,
    lab_voxel_size,
    reciprocal_basis,
    resample_to_lab,
    wavelength,
)
from qorageom.spec import SpecScan, _normalize_motor

_OVERRIDES = {"th", "chi", "phi", "delta", "gamma", "energy", "distance_m", "chi_offset",
              "scanmot", "scanmot_del"}


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


def _missing(scan: SpecScan, what: str, override: str):
    return ValueError(f"{what} not found in scan {scan.scan_no}; pass {override}=... to set it")


def geometry_from_spec(
    scan: SpecScan, diffractometer="34idc", detector=None, shape=None, binning=(1, 1, 1),
    **overrides,
) -> ScanGeometry:
    """Geometry of a rocking scan from the values in its SPEC header.

    Keyword overrides win over the file, as in cohere: the circle angles in degrees by
    mnemonic (``th``, ``chi``, ``phi``, ``delta``, ``gamma``), ``energy`` (keV or eV),
    ``distance_m``, ``chi_offset`` (degrees, default 90 for the 34-ID-C chi motor), and
    ``scanmot`` with ``scanmot_del`` (step per frame in the motor's unit) when the scan command
    is not an ``ascan``/``dscan``. ``detector`` defaults to ``#UIMDET``. ``shape`` is the
    (N1, N2, N3) of the volume, ``binning`` the detector binning of the data and the number of
    frames averaged per rocking step.

    Theta, phi and energy rocking are supported; only the ``34idc`` diffractometer so far. The
    columns follow the detector orientation of the registry, so a ``'y-'`` axis flips its column.
    """
    unknown = set(overrides) - _OVERRIDES
    if unknown:
        raise TypeError(f"unknown override(s) {sorted(unknown)}, known: {sorted(_OVERRIDES)}")
    diff = get_diffractometer(diffractometer)
    if diff.name != "34idc":
        raise NotImplementedError(f"geometry_from_spec supports '34idc' only, not {diff.name!r}")

    det_name = detector or scan.detector
    if det_name is None:
        raise _missing(scan, "detector (#UIMDET)", "detector")
    det = get_detector(det_name)
    if shape is None:
        raise ValueError("pass shape=(N1, N2, N3), the shape of the volume to describe")
    orientation = tuple(o[0] for o in det.orientation)
    if orientation != ("x", "y"):
        raise NotImplementedError(f"detector orientation {det.orientation} is not supported")

    angles = {}
    for mne in ("th", "chi", "phi", "delta", "gamma"):
        if mne in overrides:
            angles[mne] = overrides[mne]
            continue
        try:
            angles[mne] = scan.motor(mne)
        except KeyError:
            raise _missing(scan, f"motor {mne!r}", mne) from None

    energy_raw = overrides.get("energy", scan.motor("energy") if _has(scan, "energy") else None)
    if energy_raw is None:
        raise _missing(scan, "energy", "energy")
    distance = overrides.get("distance_m", scan.detector_distance_m)
    if distance is None:
        raise _missing(scan, "detector distance (camdist or sdd)", "distance_m")

    axis = scan.scan_axis
    if "scanmot" in overrides or "scanmot_del" in overrides:
        motor = overrides.get("scanmot", axis.motor if axis else None)
        step = overrides.get("scanmot_del", axis.step if axis else None)
    else:
        motor, step = (axis.motor, axis.step) if axis else (None, None)
    if motor is None or step is None:
        raise _missing(scan, "scan axis (the command is not an ascan or dscan)", "scanmot and scanmot_del")
    rocking = _normalize_motor(motor)
    step = step * binning[2]

    pixel = (det.pixel_size[0] * binning[0], det.pixel_size[1] * binning[1])
    energy_kev, step_kev = energy_to_kev(energy_raw, step)
    lam = wavelength(energy_kev) * 1e-10
    gamma, delta = np.radians([angles["gamma"], angles["delta"]])
    if rocking in ("theta", "phi"):
        step_rad = np.radians(step)
        B = reciprocal_basis(
            lam, step_rad, distance, gamma, delta, pixel[0],
            theta=np.radians(angles["th"]), chi=np.radians(angles["chi"]), axis=rocking,
            chi_offset=np.radians(overrides.get("chi_offset", 90.0)),
        )
        rock_step = step_rad
    elif rocking == "energy":
        B = energy_basis(energy_kev, step_kev, distance, gamma, delta, pixel[0])
        # the edge lam / (rock_step N3) of lab_voxel_size is 1 / (|q_k| N3) here
        rock_step = lam * np.linalg.norm(B[:, 2])
    else:
        raise ValueError(f"cannot rock about {motor!r}; supported: theta, phi, energy")
    B[:, 1] *= pixel[1] / pixel[0]
    B[:, 0] *= 1 if det.orientation[0].endswith("+") else -1
    B[:, 1] *= 1 if det.orientation[1].endswith("+") else -1

    shape = tuple(int(n) for n in shape)
    return ScanGeometry(
        diffractometer=diff.name,
        detector=det.name,
        axis=rocking,
        shape=shape,
        wavelength_m=lam,
        energy_kev=energy_kev,
        distance_m=distance,
        B=B,
        A=direct_basis(B, shape),
        voxel_size=lab_voxel_size(lam, distance, pixel, shape, rock_step),
        diagnostics=basis_diagnostics(B),
    )


def _has(scan: SpecScan, name: str) -> bool:
    try:
        scan.motor(name)
    except KeyError:
        return False
    return True
