"""The geometry core must not depend on the file adapters (SPEC parser, frame loaders).

Core modules compute from numbers; ``qorageom.adapters`` reads file formats. A core module that
imports an adapter makes every geometry user load a parser. The one allowed exception is a
module ``__getattr__`` that forwards a deprecated name to its new home in the adapters.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "qorageom"
CORE = ["__init__", "geometry", "diffractometer", "detector", "strain", "scan_geometry"]
ADAPTERS = {"qorageom.adapters", "qorageom.adapters.spec", "qorageom.adapters.frames"}
OLD_PATHS = {"qorageom.spec", "qorageom.io"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = {
        id(n)
        for f in tree.body
        if isinstance(f, ast.FunctionDef) and f.name == "__getattr__"
        for n in ast.walk(f)
    }
    names = set()
    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
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


@pytest.mark.parametrize("module", CORE)
def test_core_module_imports_no_adapter(module):
    assert not imported_modules(PACKAGE / f"{module}.py") & (ADAPTERS | OLD_PATHS)


def loaded_after(statement: str) -> set[str]:
    code = f"import sys\n{statement}\nprint('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


@pytest.mark.parametrize("adapter", sorted(ADAPTERS | OLD_PATHS))
def test_import_qorageom_does_not_load_adapter(adapter):
    assert adapter not in loaded_after("import qorageom")
