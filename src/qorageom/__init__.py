from qorageom.detector import DETECTORS, Detector, get_detector
from qorageom.diffractometer import DIFFRACTOMETERS, Diffractometer, get_diffractometer
from qorageom.geometry import (
    HC_KEV_ANGSTROM,
    basis_diagnostics,
    detector_pose,
    direct_basis,
    energy_basis,
    energy_to_kev,
    lab_voxel_size,
    pixel_to_q_lab,
    q_to_tth,
    reciprocal_basis,
    resample_to_lab,
    rotate,
    rotation_matrix,
    shear_correction,
    tth_map,
    tth_to_d,
    tth_to_q,
    wavelength,
)
from qorageom.scan_geometry import ScanGeometry, rocking_scan_geometry
from qorageom.strain import displacement_gradient, strain

__all__ = [
    "DETECTORS",
    "DIFFRACTOMETERS",
    "HC_KEV_ANGSTROM",
    "Detector",
    "Diffractometer",
    "ScanGeometry",
    "basis_diagnostics",
    "detector_pose",
    "direct_basis",
    "displacement_gradient",
    "energy_basis",
    "energy_to_kev",
    "get_detector",
    "get_diffractometer",
    "lab_voxel_size",
    "pixel_to_q_lab",
    "q_to_tth",
    "reciprocal_basis",
    "resample_to_lab",
    "rocking_scan_geometry",
    "rotate",
    "rotation_matrix",
    "shear_correction",
    "strain",
    "tth_map",
    "tth_to_d",
    "tth_to_q",
    "wavelength",
]


def __getattr__(name):
    # qorageom.geometry_from_spec stays importable while callers move to the SPEC adapter
    if name == "geometry_from_spec":
        import warnings

        from qorageom.adapters import spec

        warnings.warn(
            "qorageom.geometry_from_spec moved to qorageom.adapters.spec.geometry_from_spec",
            DeprecationWarning,
            stacklevel=2,
        )
        return spec.geometry_from_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
