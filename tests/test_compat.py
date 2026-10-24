"""Import paths from before the adapter split keep working, with a DeprecationWarning."""

import importlib
import sys

import pytest

import qorageom
from qorageom.adapters import frames, spec


def fresh_import(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize(
    ("old", "new", "names"),
    [
        (
            "qorageom.spec",
            spec,
            ["Spec", "SpecScan", "ScanAxis", "geometry_from_spec", "_xml_to_dict"],
        ),
        ("qorageom.io", frames, ["load_mat", "load_tif_stack", "load_h5", "dump_h5"]),
    ],
)
def test_old_module_forwards_to_the_adapter(old, new, names):
    with pytest.warns(DeprecationWarning, match="qorageom.adapters"):
        module = fresh_import(old)
    for name in names:
        assert getattr(module, name) is getattr(new, name)


def test_from_import_of_an_old_module_works():
    sys.modules.pop("qorageom.spec", None)
    with pytest.warns(DeprecationWarning):
        from qorageom.spec import Spec
    assert Spec is spec.Spec


@pytest.mark.parametrize("owner", ["qorageom", "qorageom.scan_geometry"])
def test_geometry_from_spec_at_its_old_place(owner):
    module = importlib.import_module(owner)
    with pytest.warns(DeprecationWarning, match="qorageom.adapters.spec.geometry_from_spec"):
        function = module.geometry_from_spec
    assert function is spec.geometry_from_spec


def test_unknown_names_still_raise():
    with pytest.raises(AttributeError):
        qorageom.no_such_name  # noqa: B018
