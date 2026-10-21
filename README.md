# QoraGeom

[![Tests](https://github.com/NaNIsaac/QoraGeom/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraGeom/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**X-ray diffraction geometry, SPEC parsing and detector-frame loaders for Python.**

QoraGeom handles the geometry and file-reading steps of analysing X-ray
diffraction data from area detectors. It converts between photon energy,
wavelength, 2θ, |q| and d-spacing, computes the true 2θ and lab-frame q of every
pixel of a flat detector on a tilted arm, builds the BCDI reciprocal-space basis
for a rocking scan, reads SPEC data files and loads detector frames from .mat,
TIFF and HDF5.

## Install

Requires Python 3.13+ with numpy and scipy; h5py, tifffile and pandas are optional extras. Not on PyPI; install from git.

**1. Install into a uv project, then check the import**

If you have no project yet, `uv init` makes one. The last command exits non-zero if the install is broken.

<!-- test -->
```bash
uv init --no-workspace --python 3.13 qorageom-demo
cd qorageom-demo
uv add "qorageom @ git+https://github.com/NaNIsaac/QoraGeom"
uv run python -c "from qorageom import wavelength; assert abs(wavelength(8.04) - 1.54209) < 1e-4"
```

With the optional loaders instead:

```bash
uv add "qorageom[h5,tiff,pandas] @ git+https://github.com/NaNIsaac/QoraGeom"
```

## Usage

```python
import numpy as np
from qorageom import wavelength, tth_to_q, tth_to_d, q_to_tth, shear_correction

lam = wavelength(8.04)  # 1.54209
tth_to_q(30.0, lam)  # 2.10909
tth_to_d(30.0, lam)  # 2.97909
q_to_tth(2.0, lam)  # 28.4147

B = shear_correction(
    wavelength(5.55) * 1e-10, np.radians(0.8 / 120), 0.511, 0.0, np.radians(69), 5.5e-5
)
# [[ 172662.33       0.     334222.41]
#  [      0.     481801.82       0.  ]
#  [-449800.75       0.     486236.02]]
```

```python
from qorageom import reciprocal_basis

# phi rocking at theta = -71.35°, chi = 85.26° (chi motor offset 90°), 10 keV
B = reciprocal_basis(
    wavelength(10.0) * 1e-10,
    np.radians(0.01),
    1.0,
    np.radians(-1.0),
    np.radians(37.0),
    5.5e-5,
    theta=np.radians(-71.35),
    chi=np.radians(85.26),
    axis="phi",
)
# [[ 354278.6388    4659.2358  284512.2546]
#  [      0.      443537.3534   73814.7868]
#  [-266968.1031    6183.0148  843526.5203]]
```

```python
from qorageom import energy_basis

# energy scan 9000 -> 9003 eV per step; the unit (keV or eV) is detected from the value
B_energy = energy_basis(
    9000.0, 3.0, 0.5, np.radians(-1.0), np.radians(37.0), 55e-6
)  # D, pixel in m; q in 1/m
# [[ 637701.5498    8386.6245 1455967.8687]
#  [      0.      798367.2361  -42228.9453]
#  [-480542.5856   11129.4266 -487528.5606]]
```

```python
from qorageom import basis_diagnostics, direct_basis

A = direct_basis(B, shape=(64, 64, 32))  # voxel vectors [m] of the FFT grid, B: the phi scan above
# [[ 3.5177e-08 -1.8551e-09  2.2293e-08]
#  [-2.0437e-10  3.5282e-08 -6.4659e-10]
#  [-1.1847e-08 -2.4617e-09  2.9584e-08]]
d = basis_diagnostics(B)
d["angles_deg"]  # [ 90.     108.2963  84.311 ]
d["cond"]  # 2.2049
```

```python
from qorageom import lab_voxel_size, resample_to_lab

lam, step = wavelength(10.0) * 1e-10, np.radians(0.01)
voxel = lab_voxel_size(lam, 1.0, 5.5e-5, (64, 64, 32), step)  # 2.4656e-08 m
B = reciprocal_basis(
    lam,
    step,
    1.0,
    np.radians(-1.0),
    np.radians(37.0),
    5.5e-5,
    theta=np.radians(-71.35),
    chi=np.radians(85.26),
    axis="phi",
)
lab = resample_to_lab(np.random.default_rng(0).random((64, 64, 32)), B, voxel)
lab.shape  # (64, 64, 32)
```

```python
from qorageom import get_diffractometer

diff = get_diffractometer("34idc")
diff.sample_axes, diff.detector_motors  # (('y+', 'z-', 'y+'), ('delta', 'gamma'))
diff.detector_rotation([37.0, -1.0]) @ [0, 0, 1]  # [ 0.6017 -0.0175  0.7985]
```

```python
from qorageom import strain

x = np.indices((12, 12, 12))[0] * 0.5  # nm
wrapped = np.angle(np.exp(1j * 10.0 * 0.02 * x))  # G = 10 rad/nm along x, strain 0.02
strain(wrapped, [10.0, 0, 0], voxel_size=0.5)[6, 6, 6]  # 0.020000000000000007
```

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
- `energy_basis(energy, delta_energy, D, gamma, delta, pixel_size)`, `energy_to_kev`: the basis for an energy scan at a fixed sample angle (lengths in m, q in 1/m); energies of 1000 or more are taken as eV.
- `direct_basis(B, shape=None)`: real-space basis conjugate to a reciprocal basis (`B.T @ A == I`, no 2π); with `shape` the FFT voxel vectors. Raises `ValueError` for a singular basis.
- `lab_voxel_size(wavelength, D, pixel_size, shape, delta_rock, fov_min=0.7)`: isotropic lab-frame voxel edge for a detector-frame volume, shrinking the field of view by at most `fov_min`.
- `resample_to_lab(volume, B, voxel_size, order=1)`: resample a volume indexed (i, j, k) like the columns of `B` onto an orthogonal lab grid of the same shape (`scipy.ndimage.map_coordinates`).
- `basis_diagnostics(B)`: column lengths, pairwise angles [deg] and condition number of a basis.

`qorageom.diffractometer`: `Diffractometer` (circle axes in xrayutilities notation such as `'z-'`, outer to inner, with SPEC motor names; `sample_rotation`, `detector_rotation` take one angle [deg] per circle; `rocking_basis` gives the reciprocal basis of a scan rocking one sample circle or the energy), `DIFFRACTOMETERS` and `get_diffractometer(name)` for `34idc`, `P10sixc` and `id01`.

`qorageom.strain`:

- `displacement_gradient(phase, voxel_size, axis)`: gradient of a wrapped phase [rad per length], robust against 2π jumps (smallest of three phase offsets per voxel, as in the legacy MATLAB strain script),
- `strain(phase, q_vector, voxel_size, support=None)`: strain along the scattering vector `q_vector` (rad per length, with the 2π), `Ĝ . grad(phase) / |G|`; nan outside `support`.

`qorageom.detector`: `Detector` (pixel size [m], axis orientation, frame shape), `DETECTORS` and `get_detector(name)` for `34idcTIM1`, `34idcTIM2`, `ASI`, `mpxgaas` and `e4m`.

`qorageom.scan_geometry`: `geometry_from_spec(scan, diffractometer="34idc", detector=None, shape=None, binning=(1, 1, 1), **overrides)` and the `ScanGeometry` it returns.

`qorageom.spec`: `Spec`, `SpecScan`. Reads `#O/#P` motor positions, `#D` time, `#L` columns and data, and `#UXML` metadata (`scan.sdd`, `scan.rois`, `scan.beam_center`). `scan.scan_axis` (motor, start, stop, intervals, step of an `ascan`/`dscan`, else None), `scan.motor(name)`, `scan.detector` (`#UIMDET`) and `scan.roi` (`#UIMR5`), `scan.energy_kev` and `scan.detector_distance_m` (`camdist` or `sdd`, mm to m). `scan.to_dataframe()` needs pandas.

`qorageom.io`: `load_mat` (.mat `data` (y, x, frame) → (frame, y, x)), `load_tif_stack`, `load_h5`, `dump_h5` (print every group/dataset with shape, dtype and attributes).

## From a SPEC scan to a lab-frame volume

`geometry_from_spec` reads the circle angles, energy, detector name and distance, and the rocking motor from a scan header, builds the basis from the diffractometer's axis strings (`Diffractometer.rocking_basis`, the same construction as xrayutilities' `QConversion`), and returns a `ScanGeometry` with the bases, the lab voxel size and the basis diagnostics. It covers every layout in the registry (`34idc`, `P10sixc`, `id01`) for rocking about any one sample circle or the energy. Keyword overrides (`delta=`, `energy=`, `distance_m=`, `detector=`, `chi_offset=`, ...) win over the file, and a missing value raises an error that names the keyword to pass. `examples/spec_to_lab.py` writes a one-scan SPEC file and resamples a volume:

```bash
uv run python examples/spec_to_lab.py
```

```
phi 34idcTIM2 9.0 keV 0.5 m
voxel 19.57 nm, cond 1.41
(64, 64, 32)
```

## From diffraction pattern to strain

`examples/bcdi_pipeline.py` simulates a 16-voxel crystal (5 nm voxels, Au 111) with a localized Gaussian lattice displacement, computes its diffraction intensity, reconstructs it with [QoraPhase](https://github.com/NaNIsaac/QoraPhase) (4 random starts), aligns it to the amplitude of the true object, removes the phase ramp with `qoraphase.remove_ramp` and recovers the strain with `qorageom.strain` on the support eroded by one voxel. The crystal is simulated directly in lab axes, so no resampling is needed; the QoraPhase dependency lives in the `examples` group.

```bash
uv run --group examples python examples/bcdi_pipeline.py
```

```
reconstruction: 13 s, Fourier error 1.8e-07
true strain:  rms 1.77e-03, range [-4.14e-03, 4.76e-03]
estimator floor on the true phase: rms 6.75e-05
strain error: rms 4.29e-04, max 6.41e-04
```

The "floor" line is the same estimator applied to the true phase, so it is the part of the error that comes from the 5 nm voxels and finite differences. The data are noiseless, and the global phase, translation and twin branch are fixed from the true amplitude.

## Tests

```bash
uv run pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
