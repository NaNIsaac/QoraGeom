# Project boundary


**Current implementation:** QoraGeom 0.1.0 contains diffraction geometry, coordinates and reciprocal-space utilities **and** legacy SPEC parsing and optional MAT/TIFF/HDF5 frame loaders. The guides describe the code available today.

**Planned boundary:** the geometry/numerical core stays independent of experiment-file formats. SPEC and detector-frame readers move to [CXD](https://github.com/NaNIsaac/CXD) in a later release; their current usage is collected in the [file-reader guide](file-readers.md).

This boundary is an architectural direction, **not** a claim that the file readers have already been removed.

