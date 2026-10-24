# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First version, installed from git so far. It holds the photon-energy, 2θ, |q| and d conversions,
detector pixel geometry, the BCDI reciprocal basis for theta, phi and energy scans, `rocking_scan_geometry`
(the bases, voxel size and diagnostics of a rocking scan from its circle readings), the
diffractometer and detector registries, and strain from a wrapped phase.

The SPEC scan parser, `geometry_from_spec` and the `.mat`, TIFF and HDF5 frame loaders that were on
main before the release moved to CXD (`cxd.io`); QoraGeom reads no file format.
