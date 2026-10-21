"""Detector pixel sizes and orientations, in the notation of cohere's beamline modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    """``pixel_size`` [m] and ``orientation`` give the direction in which array axis 0 and axis 1
    advance in the detector plane, in xrayutilities notation; ``dims`` is the full frame shape.
    """

    name: str
    pixel_size: tuple[float, float]
    orientation: tuple[str, str]
    dims: tuple[int, int]


DETECTORS = {
    d.name: d
    for d in (
        Detector("34idcTIM1", (55e-6, 55e-6), ("x+", "y-"), (256, 256)),
        Detector("34idcTIM2", (55e-6, 55e-6), ("x+", "y-"), (512, 512)),
        Detector("ASI", (55e-6, 55e-6), ("x+", "y-"), (518, 518)),
        Detector("mpxgaas", (55e-6, 55e-6), ("x-", "y-"), (516, 516)),
        Detector("e4m", (75e-6, 75e-6), ("x+", "y-"), (2167, 2070)),
    )
}


def get_detector(name: str) -> Detector:
    try:
        return DETECTORS[name]
    except KeyError:
        raise KeyError(f"Unknown detector {name!r}, known: {sorted(DETECTORS)}") from None
