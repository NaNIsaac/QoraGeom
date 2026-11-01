# Installation

QoraGeom 0.1.0 is available from [PyPI](https://pypi.org/project/qorageom/).
It requires Python 3.13 or later and installs NumPy and SciPy.
Inside an existing uv project, use `uv add qorageom`.

To start a separate project with all optional file readers:

<!-- test -->
```sh
uv init --no-workspace --python 3.13 qorageom-extras-demo
uv add --directory qorageom-extras-demo "qorageom[h5,tiff,pandas]"
uv run --directory qorageom-extras-demo python -c "import qorageom, h5py, tifffile, pandas"
```

Choose only the extras you need: `h5` for HDF5, `tiff` for TIFF, and `pandas` for
`SpecScan.to_dataframe()`. Geometry operations need none of these extras.

For development, clone the repository and run `uv run --group dev pytest -q tests`.
The BCDI reconstruction example additionally uses the `examples` dependency group.
