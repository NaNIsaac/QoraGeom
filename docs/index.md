# QoraGeom

QoraGeom does the geometry and file-reading steps that come before analysing X-ray diffraction data
from an area detector. It converts between photon energy, wavelength, 2θ, |q| and d-spacing,
computes the lab-frame q of every pixel of a detector on a tilted arm, builds the reciprocal-space
basis of a BCDI rocking scan, reads SPEC scan files, and loads detector frames from `.mat`, TIFF
and HDF5. [CXD](https://github.com/NaNIsaac/CXD) and
[insitu-heating-xrd](https://github.com/NaNIsaac/insitu-heating-xrd) build on it.

It does not reconstruct anything: phase retrieval is [QoraPhase](https://github.com/NaNIsaac/QoraPhase).

## Install

Python 3.13 or later. QoraGeom is not on PyPI yet; install it from git into a uv project:

```sh
uv add "qorageom @ git+https://github.com/NaNIsaac/QoraGeom"
```

h5py, tifffile and pandas are optional extras (`qorageom[h5,tiff,pandas]`).

## First run

```python
from qorageom import q_to_tth, tth_to_d, tth_to_q, wavelength

lam = wavelength(8.04)  # photon energy in keV, wavelength in Å
print(f"{lam:.5f}")
print(f"{tth_to_q(30.0, lam):.5f}")
print(f"{tth_to_d(30.0, lam):.5f}")
print(f"{q_to_tth(2.0, lam):.4f}")
```

Output, from qorageom 0.1.0 on Python 3.13 (Windows):

```
1.54209
2.10909
2.97909
28.4147
```

The values are λ in Å, |q| in 1/Å, d in Å and 2θ in degrees.

## Where to go next

The README is the full guide. Its sections:

- [Usage](../README.md#usage): reciprocal bases for theta, phi and energy scans, `direct_basis`, `resample_to_lab`, diffractometers, strain from a wrapped phase and the SPEC parser, each with its output.
- [Modules](../README.md#modules): every public function by module.
- [From a SPEC scan to a lab-frame volume](../README.md#from-a-spec-scan-to-a-lab-frame-volume): `rocking_scan_geometry`, `geometry_from_spec` and `ScanGeometry`.
- [From diffraction pattern to strain](../README.md#from-diffraction-pattern-to-strain).
- [Tests](../README.md#tests) and [Citation](../README.md#citation).

The Python API page on this site lists the public functions and classes with their docstrings.
