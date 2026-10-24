"""Detector frame loaders (an adapter): MATLAB .mat stacks, TIFF frames, HDF5 datasets.

h5py and tifffile are optional, so they are imported inside the functions that use them.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.io import loadmat

__all__ = ["dump_h5", "load_h5", "load_mat", "load_tif_stack"]


def load_mat(file: str | Path, key: str = "data") -> np.ndarray:
    """Read a (y, x, frame) array from a .mat file and return it as (frame, y, x)."""
    mat = loadmat(str(file))
    if key not in mat:
        raise KeyError(f"'{key}' key not found in the .mat file: {file}")
    return np.transpose(mat[key], (2, 0, 1))


def load_tif_stack(files: Iterable[str | Path]) -> np.ndarray:
    """Stack single-frame TIFF files into one (frame, y, x) array, in the given order."""
    import tifffile

    return np.stack([tifffile.imread(f) for f in files])


def load_h5(file: str | Path, dataset: str) -> np.ndarray:
    """Read one dataset from an HDF5 file.

    Import hdf5plugin first for bitshuffle/LZ4-compressed data (PAL-XFEL, Eiger).
    """
    import h5py

    with h5py.File(file, "r") as f:
        return f[dataset][()]


def dump_h5(file: str | Path) -> None:
    """Print every group/dataset in an HDF5 file with its shape, dtype and attributes.

    Only metadata is read, so compressed datasets need no filter plugin.
    """
    import h5py

    def show(name: str, obj) -> None:
        match obj:
            case h5py.Dataset():
                print(f"{name}  {obj.shape} {obj.dtype}")
            case _:
                print(f"{name}/")
        for key, value in obj.attrs.items():
            print(f"    @{key} = {value}")

    with h5py.File(file, "r") as f:
        f.visititems(show)
