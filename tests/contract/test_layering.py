"""Architecture guard: a layer may import only from layers at or below it."""

from __future__ import annotations

import ast
from pathlib import Path

import horasis

# Higher number = higher layer. `viz` is a drawing utility that depends only
# on the foundation, so it sits alongside it.
LAYER_RANK = {
    "foundation": 1,
    "viz": 1,
    "backend": 2,
    "plugins": 2,
    "orchestration": 3,
    "access": 4,
    "api": 5,
}


def _horasis_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = node.names[0].name.split(".")
        else:
            continue
        if parts[0] == "horasis" and len(parts) > 1:
            found.add(parts[1])
    return found


def test_layers_only_import_downward() -> None:
    root = Path(horasis.__file__).parent
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        own = rel.parts[0].removesuffix(".py")
        if own not in LAYER_RANK:
            continue
        for target in _horasis_imports(path):
            if target in LAYER_RANK and LAYER_RANK[target] > LAYER_RANK[own]:
                violations.append(f"{rel} ({own}) imports horasis.{target}")
    assert not violations, "\n".join(violations)
