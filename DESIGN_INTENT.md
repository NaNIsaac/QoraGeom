# Design intent

## Purpose

QoraGeom is the X-ray diffraction geometry library: energy, wavelength, 2θ, |q| and d-spacing,
the lab-frame q of a detector pixel, and the reciprocal-space basis of a BCDI scan. It computes
from numbers and does not read experiment files or reconstruct data.

## Responsibilities

- `geometry`: unit conversions, the per-pixel 2θ map, `pixel_to_q_lab`, detector pose, rotations,
  reciprocal and direct bases with their diagnostics, lab voxel size, resampling to the lab frame
  and the shear correction.
- `diffractometer` and `detector`: the instrument models and their presets (`DIFFRACTOMETERS`,
  `DETECTORS`), including the motor names of each diffractometer.
- `scan_geometry`: `rocking_scan_geometry` and the `ScanGeometry` result it builds from plain
  angles and instrument models.
- `strain`: displacement gradient and strain from a reconstructed phase.

## Planned boundary

QoraGeom becomes geometry only. The SPEC parser (`qorageom.spec`), the frame loaders
(`qorageom.io`) and `geometry_from_spec`, the bridge from a SPEC scan to `ScanGeometry`, move to
CXD's reader layer, as the [project boundary guide](docs/guides/boundary.md) says. No
compatibility re-exports are planned, and the `h5`, `tiff` and `pandas` extras leave with the
readers. Until the move they are still here (see Known gaps).

## Non-goals

- Reading SPEC files, detector frames or HDF5 under the planned boundary (see above).
- Phase retrieval: QoraPhase.
- Scan analysis, q-space regridding and viewers: CXD.
- Plotting of any kind.

## Dependencies

Runtime: numpy and scipy only. The core modules (`geometry`, `diffractometer`, `detector`,
`strain`, `scan_geometry`) must never import `qorageom.spec` or `qorageom.io`, so a geometry user
does not load a file parser. No GUI, plotting or file-format library is imported by the core.

h5py, tifffile and pandas are optional extras used only by the readers, imported inside the
function that needs them. `qoraphase` is in the `examples` group only.

`tests/test_import_boundary.py` checks that no core module imports a reader and that
`import qorageom` loads neither `qorageom.spec` nor `qorageom.io`. Known violations are
`xfail(strict=True)`, so a fix turns the test red until the mark goes.

## Boundaries to sibling libraries

- CXD: the reader layer (SPEC parser, `geometry_from_spec`, frame loaders) under the planned
  boundary, and everything that analyses a scan. CXD imports QoraGeom; QoraGeom never imports CXD.
- QoraPhase: phase retrieval. QoraGeom supplies the reciprocal basis that maps a detector-frame
  reconstruction to lab or sample axes.
- splotter: has its own SPEC parser, separate from `qorageom.spec` (see Known gaps).

## Conventions

- Scalar conversions take 2θ in degrees and give |q| in 1/Å with 2π included (`tth_to_q`);
  `pixel_to_q_lab` gives q in 1/Å with 2π included.
- The basis functions (`reciprocal_basis`, `direct_basis` and the others built on them) take
  angles in radians and give q without 2π, in the inverse of the length unit passed in. Because
  the two families differ, a function's docstring states its units.
- `HC_KEV_ANGSTROM` is the CODATA 2018 value of hc.
- Motor names belong to the `Diffractometer` model, next to the geometry they drive.
- `tests/data/scan_geometry_golden.json` pins `geometry_from_spec` output; a change to the numbers
  is a physics change and is stated in the pull request.
- `tests/test_cross_implementations.py` compares the geometry with other implementations on the
  same inputs; a disagreement is a strict xfail with its reason.

## Criteria for a new feature

- It is diffraction geometry computed from numbers, with no file format inside.
- Runtime dependencies stay numpy and scipy.
- It has a test against a closed form or an independent implementation.
- Units are written in the docstring, and a new instrument preset says where its axes and signs
  come from.
- A change to physical constants, conventions or numbers is stated in the pull request.

## Known gaps

- Readers not moved yet: `qorageom.spec`, `qorageom.io` and `geometry_from_spec` are still in this
  package, with the `h5`, `tiff` and `pandas` extras.
- `scan_geometry` imports `qorageom.spec`, because `geometry_from_spec` takes a `SpecScan`, so
  `import qorageom` loads the parser (the strict xfails in `tests/test_import_boundary.py`).
- `qorageom.spec` imports the private `_normalize_motor` from `diffractometer`, a private import
  that would cross packages once the parser is in CXD.
- splotter keeps a second SPEC parser of its own, apart from `qorageom.spec`.
