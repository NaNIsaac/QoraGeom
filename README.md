# QoraGeom

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**X-ray diffraction geometry, SPEC parsing and detector-frame loaders for Python.**

QoraGeom handles the geometry and file-reading steps of analysing X-ray
diffraction data from area detectors. It converts between photon energy,
wavelength, 2θ, |q| and d-spacing, computes the true 2θ and lab-frame q of every
pixel of a flat detector on a tilted arm, builds the BCDI reciprocal-space basis
for a rocking scan, reads SPEC data files and loads detector frames from .mat,
TIFF and HDF5.

## Install

Requires Python 3.13+ with numpy and scipy; h5py, tifffile and pandas are optional extras. Not on PyPI; install from git:

```bash
uv add "qorageom @ git+https://github.com/NaNIsaac/QoraGeom"
uv add "qorageom[h5,tiff,pandas] @ git+https://github.com/NaNIsaac/QoraGeom"   # with the optional loaders
```

## Usage

```python
import numpy as np
from qorageom import wavelength, tth_to_q, tth_to_d, q_to_tth, shear_correction

lam = wavelength(8.04)        # 1.54209
tth_to_q(30.0, lam)           # 2.10909
tth_to_d(30.0, lam)           # 2.97909
q_to_tth(2.0, lam)            # 28.4147

B = shear_correction(wavelength(5.55) * 1e-10, np.radians(0.8 / 120), 0.511, 0.0, np.radians(69), 5.5e-5)
# [[ 172662.33       0.     334222.41]
#  [      0.     481801.82       0.  ]
#  [-449800.75       0.     486236.02]]
```

```python
from qorageom import reciprocal_basis

# phi rocking at theta = -71.35°, chi = 85.26° (chi motor offset 90°), 10 keV
B = reciprocal_basis(wavelength(10.0) * 1e-10, np.radians(0.01), 1.0, np.radians(-1.0), np.radians(37.0), 5.5e-5,
                     theta=np.radians(-71.35), chi=np.radians(85.26), axis="phi")
# [[ 354278.6388    4659.2358  284512.2546]
#  [      0.      443537.3534   73814.7868]
#  [-266968.1031    6183.0148  843526.5203]]
```

```python
from qorageom import basis_diagnostics, direct_basis

A = direct_basis(B, shape=(64, 64, 32))   # voxel vectors [m] of the FFT grid, B from above
# [[ 3.5177e-08 -1.8551e-09  2.2293e-08]
#  [-2.0437e-10  3.5282e-08 -6.4659e-10]
#  [-1.1847e-08 -2.4617e-09  2.9584e-08]]
d = basis_diagnostics(B)
d["angles_deg"]   # [ 90.     108.2963  84.311 ]
d["cond"]         # 2.2049
```

```python
from qorageom.spec import Spec

with Spec("path/to/specfile") as sp:
    scan = sp.get_scan(1)
    scan.motors        # {'Theta': 10.0, 'Delta': 69.0, 'Gamma': 0.0}
    scan.sdd           # 511.0, from #UXML <sample_detector_distance>
    scan.to_dataframe()
```

```python
# scan 1: "ascan Phi 0.5 -0.5 20 2", #UIMDET Timepix:, #UIMR5 100 120 64 64
scan.scan_axis             # ScanAxis(motor='Phi', start=0.5, stop=-0.5, intervals=20)
scan.scan_axis.step        # -0.05
scan.motor("del")          # 37.01, matches Delta ignoring case and the del/gam aliases
scan.detector, scan.roi    # ('Timepix', [100, 120, 64, 64])
```

`main.py` prints the detector pose and the q vector of pixel (1, 1) for an 8.04 keV beam, a 1 m detector distance and a 30° up angle:

```
$ uv run python main.py
center_pos:   [0.8660254 0.5       0.       ]
pixel_x_dir: [ 0.  0. -1.]
pixel_y_dir: [-0.5        0.8660254  0.       ]
q_lab:       [-5.45985588e-01  2.03742181e+00 -2.24095052e-04]
```

## Modules

`qorageom.geometry` (also exported from `qorageom`):

- `HC_KEV_ANGSTROM`, `wavelength(energy_kev)`: λ [Å] from photon energy,
- `tth_to_q`, `q_to_tth`, `tth_to_d`: 2θ [deg] ↔ |q| [1/Å] ↔ d [Å] at a given λ,
- `tth_map(shape, pixel_m, sdd_m, xc, yc, alpha_deg)`: true 2θ of every pixel of a flat detector on a tilted arm,
- `detector_pose`, `pixel_to_q_lab`: detector center/pixel directions from arm angles, and the lab-frame q of a pixel,
- `rotate`, `rotation_matrix`: Rodrigues rotation (vector and matrix form),
- `shear_correction(wavelength, delta_theta, D, gamma, delta, pixel_size)`: BCDI reciprocal basis `B_recip` (34-ID-C convention), `q = B_recip @ (i, j, k)` for detector pixel (i, j) and rocking step k.
- `reciprocal_basis(wavelength, delta_rock, D, gamma, delta, pixel_size, theta, chi, axis="theta"|"phi", chi_offset)`: the same basis for a theta or phi rocking scan with the theta/chi/phi sample circles; the k column is the exact rotation step, so it matches `shear_correction` to first order in `delta_rock`.
- `direct_basis(B, shape=None)`: real-space basis conjugate to a reciprocal basis (`B.T @ A == I`, no 2π); with `shape` the FFT voxel vectors. Raises `ValueError` for a singular basis.
- `basis_diagnostics(B)`: column lengths, pairwise angles [deg] and condition number of a basis.

`qorageom.spec`: `Spec`, `SpecScan`. Reads `#O/#P` motor positions, `#D` time, `#L` columns and data, and `#UXML` metadata (`scan.sdd`, `scan.rois`, `scan.beam_center`). `scan.scan_axis` (motor, start, stop, intervals, step of an `ascan`/`dscan`, else None), `scan.motor(name)`, `scan.detector` (`#UIMDET`) and `scan.roi` (`#UIMR5`). `scan.to_dataframe()` needs pandas.

`qorageom.io`: `load_mat` (.mat `data` (y, x, frame) → (frame, y, x)), `load_tif_stack`, `load_h5`, `dump_h5` (print every group/dataset with shape, dtype and attributes).

## Tests

```bash
uv run pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
