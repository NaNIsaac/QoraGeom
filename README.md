# QoraGeom

[![Tests](https://github.com/NaNIsaac/QoraGeom/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraGeom/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="https://raw.githubusercontent.com/NaNIsaac/QoraGeom/main/docs/media/tth_map.png" width="720" alt="2θ contours and |q| of every pixel of a flat detector, with the arm at 0° and tilted by 25°">

Per-pixel 2θ and |q| from `tth_map` for one detector at two arm tilts ([script](docs/media/make_figure.py)).

**X-ray diffraction geometry for Python.**

QoraGeom handles the geometry of analysing X-ray diffraction data from area
detectors. It converts between photon energy,
wavelength, 2θ, |q| and d-spacing, computes the true 2θ and lab-frame q of every
pixel of a flat detector on a tilted arm and builds the BCDI reciprocal-space basis
for a rocking scan. It reads no file format: SPEC files and detector frames are
read by the applications, for example [CXD](https://github.com/NaNIsaac/CXD)'s
`cxd.io`, which pass plain values to these functions.

## Project boundary (current vs planned)

**Current implementation:** QoraGeom contains diffraction geometry, coordinates, reciprocal-space and strain functions only. No module reads a file format and `import qorageom` loads no reader (`tests/test_import_boundary.py`). The SPEC parser, `geometry_from_spec` and the .mat/TIFF/HDF5 frame loaders that used to be here moved to CXD's I/O layer `cxd.io`; there are no compatibility aliases. The README examples below describe the code as it works today.

**Boundary:** the reusable geometry/numerical core stays independent of experiment-file formats; file readers belong to the I/O and application layers that use them ([CXD](https://github.com/NaNIsaac/CXD) `cxd.io` today). CXD moved its users to `cxd.io` before the readers were removed here (CXD#22).

This boundary is an architectural direction, **not** a claim that the file readers have already been removed.

## Install

Requires Python 3.13+ with numpy and scipy. Not on PyPI yet; install from git. The first release will add `pip install qorageom`.

**1. Install into a uv project, then check the import**

If you have no project yet, `uv init` makes one. The last command exits non-zero if the install is broken.

<!-- test -->
```bash
uv init --no-workspace --python 3.13 qorageom-demo
cd qorageom-demo
uv add "qorageom @ git+https://github.com/NaNIsaac/QoraGeom"
uv run python -c "from qorageom import wavelength; assert abs(wavelength(8.04) - 1.54209) < 1e-4"
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

`qorageom.scan_geometry`: `rocking_scan_geometry(shape, energy_kev, distance_m, sample_angles_deg, detector_angles_deg, rock, step, *, diffractometer, detector, binning=(1, 1, 1), sample_offsets_deg=None)` and the `ScanGeometry` it returns: the bases, lab voxel size and diagnostics of a rocking scan (angles in degrees as a sequence in motor order or a mapping by motor name; `step` in degrees per frame, or keV per frame for `rock="energy"`). `diffractometer` and `detector` take a registry name or an instance.

## From a rocking scan to a lab-frame volume

`rocking_scan_geometry` takes the circle readings, energy, detector distance, detector and the rocking motor and step, builds the basis from the diffractometer's axis strings (`Diffractometer.rocking_basis`, the same construction as xrayutilities' `QConversion`), and returns a `ScanGeometry` with the bases, the lab voxel size and the basis diagnostics. It covers every layout in the registry (`34idc`, `P10sixc`, `id01`) for rocking about any one sample circle or the energy. Reading those values from a SPEC header is `cxd.io.spec.geometry_from_spec` in CXD. `examples/scan_to_lab.py` resamples a volume:

```bash
uv run python examples/scan_to_lab.py
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
