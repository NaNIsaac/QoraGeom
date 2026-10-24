import numpy as np
import pytest
from scipy.io import savemat

from qorageom.adapters.frames import dump_h5, load_h5, load_mat, load_tif_stack


def test_load_mat_transposes_to_frame_first(tmp_path):
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (y, x, frame)
    savemat(tmp_path / "a.mat", {"data": arr})
    out = load_mat(tmp_path / "a.mat")
    assert out.shape == (4, 2, 3)
    np.testing.assert_array_equal(out[1], arr[:, :, 1])


def test_load_mat_missing_key(tmp_path):
    savemat(tmp_path / "a.mat", {"other": np.zeros((1, 1, 1))})
    with pytest.raises(KeyError, match="'data'"):
        load_mat(tmp_path / "a.mat")


def test_load_tif_stack_keeps_order(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    files = []
    for i in range(3):
        f = tmp_path / f"f{i}.tif"
        tifffile.imwrite(f, np.full((4, 5), i, dtype=np.uint16))
        files.append(f)
    out = load_tif_stack(files[::-1])
    assert out.shape == (3, 4, 5)
    assert list(out[:, 0, 0]) == [2, 1, 0]


def test_h5_load_and_dump(tmp_path, capsys):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("scan")
        g.attrs["energy"] = 9.7
        g.create_dataset("img", data=np.ones((2, 3), dtype=np.float32))

    np.testing.assert_array_equal(load_h5(path, "scan/img"), np.ones((2, 3)))

    dump_h5(path)
    assert capsys.readouterr().out.splitlines() == [
        "scan/",
        "    @energy = 9.7",
        "scan/img  (2, 3) float32",
    ]
