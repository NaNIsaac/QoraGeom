"""QoraGeom is geometry only: no module reads a file format, and importing it loads no reader.

SPEC files and detector frames are read by the applications (CXD's ``cxd.io``), which pass plain
values to ``rocking_scan_geometry`` and the other functions here.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "qorageom"
READERS = {"h5py", "pandas", "scipy.io", "tifffile", "xml.etree", "xml.etree.ElementTree"}


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.stem)
def test_module_imports_no_file_reader(path):
    assert not imported_modules(path) & READERS


def test_package_has_no_reader_modules():
    assert not {p.stem for p in PACKAGE.glob("*.py")} & {"spec", "io", "adapters"}


def test_import_qorageom_loads_no_file_reader():
    code = "import sys, qorageom\nprint('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert not set(out.split()) & READERS
