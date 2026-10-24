"""The geometry core must not depend on the file adapters (SPEC parser, frame loaders).

Core modules compute from numbers; ``spec`` and ``io`` read file formats. A core module that
imports an adapter makes every geometry user load a parser. Known violations are
``xfail(strict=True)`` so the test turns red, and has to be updated, once they are fixed.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "qorageom"
CORE = ["geometry", "diffractometer", "detector", "strain", "scan_geometry"]
ADAPTERS = {"qorageom.spec", "qorageom.io"}
SCAN_GEOMETRY_IMPORTS_SPEC = pytest.mark.xfail(
    strict=True,
    reason="geometry_from_spec takes a SpecScan, so scan_geometry imports qorageom.spec",
)


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = "qorageom" if node.module is None else f"qorageom.{node.module}"
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(m, marks=SCAN_GEOMETRY_IMPORTS_SPEC) if m == "scan_geometry" else m
        for m in CORE
    ],
)
def test_core_module_imports_no_adapter(module):
    assert not imported_modules(PACKAGE / f"{module}.py") & ADAPTERS


def loaded_after(statement: str) -> set[str]:
    code = f"import sys\n{statement}\nprint('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


@pytest.mark.parametrize(
    "adapter",
    [pytest.param("qorageom.spec", marks=SCAN_GEOMETRY_IMPORTS_SPEC), "qorageom.io"],
)
def test_import_qorageom_does_not_load_adapter(adapter):
    assert adapter not in loaded_after("import qorageom")
