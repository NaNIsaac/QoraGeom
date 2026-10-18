# QoraGeom

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Shared core for X-ray diffraction analysis: detector and scattering geometry, SPEC parsing and detector-frame loaders.**

numpy and scipy only; h5py, tifffile and pandas are optional and imported only by the functions that need them.

## Modules

`qorageom.geometry` (also exported from `qorageom`):

- `HC_KEV_ANGSTROM`, `wavelength(energy_kev)`: λ [Å] from photon energy,
- `tth_to_q`, `q_to_tth`, `tth_to_d`: 2θ [deg] ↔ |q| [1/Å] ↔ d [Å] at a given λ,
- `tth_map(shape, pixel_m, sdd_m, xc, yc, alpha_deg)`: true 2θ of every pixel of a flat detector on a tilted arm,
- `detector_pose`, `pixel_to_q_lab`: detector center/pixel directions from arm angles, and the lab-frame q of a pixel,
- `rotate`, `rotation_matrix`: Rodrigues rotation (vector and matrix form),
- `shear_correction(wavelength, delta_theta, D, gamma, delta, pixel_size)`: BCDI reciprocal basis `B_recip` (34-ID-C convention), `q = B_recip @ (i, j, k)` for detector pixel (i, j) and rocking step k.

`qorageom.spec`: `Spec`, `SpecScan`. Reads `#O/#P` motor positions, `#D` time, `#L` columns and data, and `#UXML` metadata (`scan.sdd`, `scan.rois`, `scan.beam_center`). `scan.to_dataframe()` needs pandas.

`qorageom.io`: `load_mat` (.mat `data` (y, x, frame) → (frame, y, x)), `load_tif_stack`, `load_h5`, `dump_h5` (print every group/dataset with shape, dtype and attributes).

## Install

Requires Python 3.13+. Not on PyPI; install from git:

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
from qorageom.spec import Spec

with Spec("path/to/specfile") as sp:
    scan = sp.get_scan(1)
    scan.motors        # {'Theta': 10.0, 'Delta': 69.0, 'Gamma': 0.0}
    scan.sdd           # 511.0, from #UXML <sample_detector_distance>
    scan.to_dataframe()
```

`main.py` prints the detector pose and the q vector of pixel (1, 1) for an 8.04 keV beam, a 1 m detector distance and a 30° up angle:

```
$ uv run python main.py
center_pos:   [0.8660254 0.5       0.       ]
pixel_x_dir: [ 0.  0. -1.]
pixel_y_dir: [-0.5        0.8660254  0.       ]
q_lab:       [-5.45985588e-01  2.03742181e+00 -2.24095052e-04]
```

## Tests

```bash
uv run pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
