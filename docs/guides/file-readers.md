# SPEC and detector-frame readers

These readers are available in 0.1.0 and move to CXD in a later release.

## SPEC metadata

The following snippets need your own SPEC file; replace `path/to/specfile` and the scan number.
`to_dataframe()` requires the `pandas` extra. The displayed header values describe the example
scan, not arbitrary files.

```python
from qorageom.spec import Spec

with Spec("path/to/specfile") as sp:
    scan = sp.get_scan(1)
    scan.motors  # {'Theta': 10.0, 'Delta': 69.0, 'Gamma': 0.0}
    scan.sdd  # 511, from #UXML <sample_detector_distance>
    scan.to_dataframe()
```

```python
# scan 1: "ascan Phi 0.5 -0.5 20 2", #UIMDET Timepix:, #UIMR5 100 120 64 64
scan.scan_axis  # ScanAxis(motor='Phi', start=0.5, stop=-0.5, intervals=20)
scan.scan_axis.step  # -0.05
scan.motor("del")  # 69.0, matches Delta ignoring case and the del/gam aliases
scan.detector, scan.roi  # ('Timepix', [100, 120, 64, 64])
```

## Detector frames

Use the `h5` and `tiff` extras for those file formats. These snippets need your own files:

```python
from qorageom.io import load_h5, load_mat, load_tif_stack

mat_frames = load_mat("scan.mat")
tiff_frames = load_tif_stack(["frame001.tif", "frame002.tif"])
h5_data = load_h5("scan.h5", "/entry/data")
```

MATLAB input is `(y, x, frame)` and becomes `(frame, y, x)`. TIFF files are stacked in the
supplied order. HDF5 retains the dataset's original axis order. Compressed HDF5 datasets may
require importing their filter plug-in before reading; no filter plug-in is installed by the `h5` extra.

## SPEC to a lab-frame volume


`geometry_from_spec` reads the circle angles, energy, detector name and distance, and the rocking motor from a scan header, builds the basis from the diffractometer's axis strings (`Diffractometer.rocking_basis`, the same construction as xrayutilities' `QConversion`), and returns a `ScanGeometry` with the bases, the lab voxel size and the basis diagnostics. It covers every layout in the registry (`34idc`, `P10sixc`, `id01`) for rocking about any one sample circle or the energy. Keyword overrides (`delta=`, `energy=`, `distance_m=`, `detector=`, `chi_offset=`, ...) win over the file, and a missing value raises an error that names the keyword to pass. `examples/spec_to_lab.py` writes a one-scan SPEC file and resamples a volume:

<!-- test -->
```bash
uv run python examples/spec_to_lab.py
```

```
phi 34idcTIM2 9.0 keV 0.5 m
voxel 19.57 nm, cond 1.41
(64, 64, 32)
```

