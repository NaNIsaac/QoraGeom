# Geometry examples

These examples retain the current 0.1.0 API. They are Python snippets, not independent shell commands; run them in order in a Python session. For a tested first command, start with the [tutorial](../tutorial.md).

## Units and conventions

- Photon energy is in keV for `wavelength`; its result is in Å. `tth_to_q` returns |q| in 1/Å.
- Reciprocal-basis examples use lengths in m and q in 1/m. `direct_basis` obeys `B.T @ A == I` without 2π.
- Wrapped-phase strain uses q in rad per length, including 2π. Keep voxel length units consistent with q.
- `energy_basis` treats energies of 1000 or more as eV. Motor-axis conventions depend on the diffractometer registry.

## Examples


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

