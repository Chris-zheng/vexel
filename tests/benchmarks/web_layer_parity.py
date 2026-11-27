"""Multilayer cross-sections: the browser engine of the web workbench (``web/``) against the
Python engine and the truth, on conformal stacks that are level, tilted and bent (synthetic,
thickness along the normal known exactly).

The browser engine is run through node (``npx tsx web/scripts/layer-parity.ts``); this needs
``npm ci`` in ``web/``. Prints a table; exits non-zero when the Python engine misses its
tolerance (the browser engine is reported, not judged: it is a preview).

    uv run python tests/benchmarks/web_layer_parity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from vexel.packs.sem.metrology import stack_layers
from vexel.synthetic import sem as S

ROOT = Path(__file__).resolve().parents[2]
CASES = {"level": {}, "tilt 6°": {"tilt": 6}, "bow 30 px": {"bow": 30}, "bow 60 px + tilt 4°": {"bow": 60, "tilt": 4}}
TOLERANCE = 0.5  # px, Python engine


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vexel-layer-parity-") as d:
        cases = {}
        for i, (name, kw) in enumerate(CASES.items()):
            img, tr = S.bent_layer_stack(seed=3, **kw)
            g = (np.clip(img, 0, 1) * 255).astype("<f4")  # the 8-bit grey the workbench shows
            key = f"case{i}"
            g.tofile(Path(d) / f"{key}.f32")
            r = stack_layers(g.astype(np.float64))
            cases[key] = {"label": name, "w": g.shape[1], "h": g.shape[0], "truth": list(tr["thicknesses"]),
                          "python": [ly["thickness"] for ly in r["layers"]]}  # fmt: skip
        (Path(d) / "cases.json").write_text(json.dumps(cases))
        out = subprocess.run(["npx", "tsx", "scripts/layer-parity.ts", d], cwd=ROOT / "web", capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    print("| case | engine | layers | thicknesses (px) | worst error (px) |\n|---|---|---|---|---|")
    ok = True
    for r in rows:
        c = cases[r["name"]]
        err = "wrong layer count" if r["maxErr"] is None else f"{r['maxErr']:.2f}"
        th = ", ".join(f"{t:.2f}" for t in r["thickness"])
        print(f"| {c['label']} (truth {', '.join(f'{t:g}' for t in c['truth'])}) | {r['engine']} | {r['layers']} | {th} | {err} |")
        if r["engine"] == "python" and (r["maxErr"] is None or r["maxErr"] > TOLERANCE):
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
