"""List the Chinese text of the SEM module (string literals and message templates) and
which of it has no English translation yet.

    uv run python scripts/i18n_extract.py [--missing] [--json]

Templates: an f-string's literal parts with ``{}`` where its values go, the form the
translation table (``src/vexel/locale/en.json``) uses as keys.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["src/vexel/ui/sem_window.py", "src/vexel/ui/sem_canvas.py", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src/vexel/packs/sem").glob("*.py"))]
CJK = re.compile(r"[一-鿿]")
SKIP_CALLS = {"compile"}  # regular expressions (goal words) are not shown


def keys(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in SKIP_CALLS:
            skip.update(id(a) for a in ast.walk(node))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body and isinstance(node.body[0], ast.Expr):
            skip.add(id(node.body[0].value))  # docstrings
    out: list[str] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
            for v in node.values:
                if isinstance(v, ast.Constant):
                    skip.add(id(v))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        else:
            continue
        if CJK.search(text) and text not in out:
            out.append(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", action="store_true", help="only the text without a translation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    table = json.loads((ROOT / "src/vexel/locale/en.json").read_text(encoding="utf-8")) if (ROOT / "src/vexel/locale/en.json").exists() else {}
    found = {}
    for f in FILES:
        for k in keys(ROOT / f):
            if not args.missing or k not in table:
                found.setdefault(k, f)
    if args.json:
        print(json.dumps(found, ensure_ascii=False, indent=1))
    else:
        for k, f in found.items():
            print(f"{f}\t{k!r}")
        print(f"{len(found)} {'without a translation' if args.missing else 'strings'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
