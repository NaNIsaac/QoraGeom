"""Diffractometer circle layouts in xrayutilities axis notation ('y+', 'z-', ...)."""
from dataclasses import dataclass

import numpy as np

from qorageom.geometry import rotation_matrix

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def axis_vector(axis: str) -> np.ndarray:
    """Rotation axis of an ``'x+'``-style string: a positive angle turns about this vector."""
    if len(axis) != 2 or axis[0] not in _AXES or axis[1] not in "+-":
        raise ValueError(f"axis must look like 'y+' or 'z-', got {axis!r}")
    return np.array(_AXES[axis[0]]) * (1 if axis[1] == "+" else -1)


@dataclass(frozen=True)
class Diffractometer:
    """Circles listed from the outermost to the innermost, motors named as in the SPEC file.

    The lab frame has the beam along ``incident_axis`` (z) and the motors zeroed at the
    reference position, so offsets such as the 34-ID-C chi offset are not included.
    """

    name: str
    sample_axes: tuple[str, ...]
    sample_motors: tuple[str, ...]
    detector_axes: tuple[str, ...]
    detector_motors: tuple[str, ...]
    incident_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        if len(self.sample_axes) != len(self.sample_motors):
            raise ValueError("sample_axes and sample_motors differ in length")
        if len(self.detector_axes) != len(self.detector_motors):
            raise ValueError("detector_axes and detector_motors differ in length")
        for axis in self.sample_axes + self.detector_axes:
            axis_vector(axis)

    def sample_rotation(self, angles_deg) -> np.ndarray:
        """Sample-to-lab rotation for one angle [deg] per sample circle."""
        return _stack(self.sample_axes, angles_deg)

    def detector_rotation(self, angles_deg) -> np.ndarray:
        """Rotation taking the zero-angle detector direction to its lab direction."""
        return _stack(self.detector_axes, angles_deg)


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
            ("y+", "z-", "y+"), ("th", "chi", "phi"),
            ("y+", "x-"), ("delta", "gamma"),
        ),
        Diffractometer(
            "P10sixc",
            ("y+", "x-", "z+", "y-"), ("mu", "om", "chi", "phi"),
            ("y+", "x-"), ("gam", "del"),
        ),
        Diffractometer(
            "id01",
            ("y-", "x-", "y-"), ("mu", "eta", "phi"),
            ("y-", "x-"), ("nu", "delta"),
        ),
    )
}


def get_diffractometer(name: str) -> Diffractometer:
    try:
        return DIFFRACTOMETERS[name]
    except KeyError:
        raise KeyError(f"Unknown diffractometer {name!r}, known: {sorted(DIFFRACTOMETERS)}") from None
