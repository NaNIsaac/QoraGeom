# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First version, installed from git so far. It holds the photon-energy, 2θ, |q| and d conversions,
detector pixel geometry, the BCDI reciprocal basis for theta, phi and energy scans, the SPEC scan
parser, the diffractometer and detector registries, strain from a wrapped phase, and the `.mat`,
TIFF and HDF5 frame loaders.

### Changed

- The SPEC parser and the frame loaders moved to `qorageom.adapters.spec` and `qorageom.adapters.frames`; the geometry core no longer imports them. `qorageom.spec`, `qorageom.io`, `qorageom.geometry_from_spec` and `qorageom.scan_geometry.geometry_from_spec` still work and raise a `DeprecationWarning`.
- `rocking_scan_geometry` builds a `ScanGeometry` from plain values; `geometry_from_spec` reads a SPEC header and calls it.
