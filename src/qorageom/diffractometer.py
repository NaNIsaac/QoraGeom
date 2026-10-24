"""Diffractometer circle layouts in xrayutilities axis notation ('y+', 'z-', ...)."""

import re
from dataclasses import dataclass

import numpy as np

from qorageom.geometry import rotation_matrix

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_MOTOR_ALIASES = {"del": "delta", "gam": "gamma", "th": "theta", "en": "energy"}


def _normalize_motor(name: str) -> str:
    """Motor name without case or punctuation, with the del/gam/th/en short forms expanded."""
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return _MOTOR_ALIASES.get(name, name)


def axis_vector(axis: str) -> np.ndarray:
    """Rotation axis of an ``'x+'``-style string: a positive angle turns about this vector."""
    if len(axis) != 2 or axis[0] not in _AXES or axis[1] not in "+-":
        raise ValueError(f"axis must look like 'y+' or 'z-', got {axis!r}")
    return np.array(_AXES[axis[0]]) * (1 if axis[1] == "+" else -1)


@dataclass(frozen=True)
class Diffractometer:
    """Circles listed from the outermost to the innermost, motors named as in the SPEC file.

    The lab frame has the beam along ``incident_axis`` (z). ``sample_offsets_deg`` is what each
    sample motor reads more than the physical circle angle (the 34-ID-C chi motor reads 90 more);
    empty means zero. ``sample_rotation`` and ``detector_rotation`` take circle angles.
    """

    name: str
    sample_axes: tuple[str, ...]
    sample_motors: tuple[str, ...]
    detector_axes: tuple[str, ...]
    detector_motors: tuple[str, ...]
    incident_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sample_offsets_deg: tuple[float, ...] = ()

    def __post_init__(self):
        if len(self.sample_axes) != len(self.sample_motors):
            raise ValueError("sample_axes and sample_motors differ in length")
        if len(self.detector_axes) != len(self.detector_motors):
            raise ValueError("detector_axes and detector_motors differ in length")
        if self.sample_offsets_deg and len(self.sample_offsets_deg) != len(self.sample_axes):
            raise ValueError("sample_offsets_deg and sample_axes differ in length")
        for axis in self.sample_axes + self.detector_axes:
            axis_vector(axis)

    def sample_rotation(self, angles_deg) -> np.ndarray:
        """Sample-to-lab rotation for one angle [deg] per sample circle."""
        return _stack(self.sample_axes, angles_deg)

    def detector_rotation(self, angles_deg) -> np.ndarray:
        """Rotation taking the zero-angle detector direction to its lab direction."""
        return _stack(self.detector_axes, angles_deg)

    def rocking_basis(
        self,
        wavelength,
        D,
        pixel_size,
        sample_motors_deg,
        detector_motors_deg,
        rock,
        step,
        orientation=("x+", "y+"),
    ):
        """Detector-frame to reciprocal-space basis ``B`` of a rocking scan, ``q = B @ (i, j, k)``.

        Motor readings are in degrees, in the order of ``sample_motors`` and ``detector_motors``.
        ``wavelength``, ``D`` and ``pixel_size`` (one value or one per detector axis) share a
        length unit and q comes out in its inverse without 2π. ``orientation`` gives the lab-
        frame detector directions that pixel axes i and j advance along at zero detector
        angles; the incident beam must be perpendicular to both.

        ``rock`` is a sample motor name, stepped by ``step`` degrees per frame, or ``"energy"``
        with ``step`` the fractional energy step dE / E. The k column is the exact change
        ``R0 R1^-1 Q - Q`` of the q a fixed detector pixel samples, with ``R0`` and ``R1`` the
        sample rotations before and after the step, as xrayutilities does for the central pixel.
        The i and j columns are the first-order change of q per pixel.
        """
        offsets = np.array(self.sample_offsets_deg or np.zeros(len(self.sample_axes)))
        sample = np.asarray(sample_motors_deg, dtype=float) - offsets
        beam = np.asarray(self.incident_axis, dtype=float)
        beam = beam / np.linalg.norm(beam)

        R_det = self.detector_rotation(detector_motors_deg)
        q0 = (R_det @ beam - beam) / wavelength
        pixel = np.broadcast_to(pixel_size, 2)
        i_dir, j_dir = (axis_vector(o) for o in orientation)
        if abs(i_dir @ beam) > 1e-12 or abs(j_dir @ beam) > 1e-12:
            raise ValueError("detector orientation must be perpendicular to the incident beam")
        qi = pixel[0] / (wavelength * D) * R_det @ i_dir
        qj = pixel[1] / (wavelength * D) * R_det @ j_dir

        if rock == "energy":
            qk = q0 * step
        else:
            if rock not in self.sample_motors:
                raise ValueError(f"cannot rock about {rock!r}; sample motors: {self.sample_motors}")
            moved = sample.copy()
            moved[self.sample_motors.index(rock)] += step
            qk = self.sample_rotation(sample) @ self.sample_rotation(moved).T @ q0 - q0
        return np.stack([qi, qj, qk], axis=1)


def _stack(axes: tuple[str, ...], angles_deg) -> np.ndarray:
    angles = np.atleast_1d(angles_deg)
    if len(angles) != len(axes):
        raise ValueError(f"expected {len(axes)} angles, got {len(angles)}")
    R = np.eye(3)
    for axis, angle in zip(axes, angles, strict=True):
        R = R @ rotation_matrix(axis_vector(axis), angle)
    return R


DIFFRACTOMETERS = {
    d.name: d
    for d in (
        Diffractometer(
            "34idc",
            ("y+", "z-", "y+"),
            ("th", "chi", "phi"),
            ("y+", "x-"),
            ("delta", "gamma"),
            sample_offsets_deg=(0.0, 90.0, 0.0),
        ),
        Diffractometer(
            "P10sixc",
            ("y+", "x-", "z+", "y-"),
            ("mu", "om", "chi", "phi"),
            ("y+", "x-"),
            ("gam", "del"),
        ),
        Diffractometer(
            "id01",
            ("y-", "x-", "y-"),
            ("mu", "eta", "phi"),
            ("y-", "x-"),
            ("nu", "delta"),
        ),
    )
}


def get_diffractometer(name: str) -> Diffractometer:
    try:
        return DIFFRACTOMETERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown diffractometer {name!r}, known: {sorted(DIFFRACTOMETERS)}"
        ) from None
