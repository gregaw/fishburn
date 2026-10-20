"""`modal` is imported in exactly one module, pipeline/exec/modal_backend.py,
so the offline tests run the same stage code as the cloud. Checked statically
(AST over every pipeline file) and at runtime (importing everything else never
pulls modal into sys.modules)."""

import ast
import importlib
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
ALLOWED = PIPELINE / "exec" / "modal_backend.py"


def modal_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name == "modal" or a.name.startswith("modal.")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "modal" or node.module.startswith("modal."):
                hits.append(node.module)
    return hits


def test_modal_imported_only_in_modal_backend():
    offenders = {
        str(p.relative_to(PIPELINE.parent)): modal_imports(p)
        for p in PIPELINE.rglob("*.py")
        if p != ALLOWED and modal_imports(p)
    }
    assert not offenders, f"modal imported outside exec/modal_backend.py: {offenders}"
    assert modal_imports(ALLOWED), "modal_backend.py should be the modal wrapper"


def test_importing_everything_else_never_loads_modal():
    for p in sorted(PIPELINE.rglob("*.py")):
        if p == ALLOWED:
            continue
        rel = p.relative_to(PIPELINE.parent).with_suffix("")
        module = ".".join(rel.parts[:-1] + (rel.parts[-1],)).replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        importlib.import_module(module)
    assert "modal" not in sys.modules
