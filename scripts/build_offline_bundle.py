"""Build an offline install bundle for computers without internet access (e.g. in a fab).

    uv run python scripts/build_offline_bundle.py --platform windows-x64 [--out dist/offline]

Minimum systems (set by PySide6 6.12's wheels): Windows 10/11 x64, macOS 14, Linux with
glibc 2.34 (e.g. Ubuntu 22.04, RHEL 9).

The bundle (a folder, then a .zip) holds everything an installation needs:

- ``vexel-<version>-py3-none-any.whl`` and ``vexel-<version>-constraints.txt`` (from uv.lock);
- ``wheels/``: every dependency as a wheel for the target platform, at the locked versions
  (including the planning-model client and the SEM text recognition, ``[agent,sem]``);
- ``uv/``: the uv binary for the target platform (the same uv version as this machine's);
- ``python/<release>/``: the Python 3.12 build uv installs (python-build-standalone);
- ``install.ps1`` / ``install.sh``: install with no network access at all;
- ``SHA256SUMS.txt`` and ``README.txt``.

Building needs internet access; installing does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLATFORMS = {
    "windows-x64": {"triple": "x86_64-pc-windows-msvc", "python_key": "windows-x86_64-none", "uv_asset": "uv-x86_64-pc-windows-msvc.zip",
                    "tags": ["win_amd64"]},
    "macos-arm64": {"triple": "aarch64-apple-darwin", "env": {"MACOSX_DEPLOYMENT_TARGET": "14.0"}, "python_key": "macos-aarch64-none", "uv_asset": "uv-aarch64-apple-darwin.tar.gz",
                    "tags": [f"macosx_{v}_0_{a}" for v in range(11, 17) for a in ("arm64", "universal2")]},
    "macos-x64": {"triple": "x86_64-apple-darwin", "env": {"MACOSX_DEPLOYMENT_TARGET": "14.0"}, "python_key": "macos-x86_64-none", "uv_asset": "uv-x86_64-apple-darwin.tar.gz",
                  "tags": [f"macosx_{v}_{m}_{a}" for v in range(10, 17) for m in (0, 9, 13, 15) for a in ("x86_64", "universal2")]},
    "linux-x64": {"triple": "x86_64-manylinux_2_34", "python_key": "linux-x86_64-gnu", "uv_asset": "uv-x86_64-unknown-linux-gnu.tar.gz",
                  "tags": ["manylinux2014_x86_64"] + [f"manylinux_2_{m}_x86_64" for m in range(17, 40)]},
}  # fmt: skip

INSTALL_PS1 = r"""# Vexel offline installation (Windows). Run in PowerShell inside this folder:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:UV_OFFLINE = "1"
$env:UV_PYTHON_INSTALL_MIRROR = ([System.Uri]"$here\python").AbsoluteUri
$uv = Join-Path $here "uv\uv.exe"
& $uv python install 3.12 --no-bin
& $uv tool install ((Join-Path $here "{wheel}") + "[agent,sem]") --python 3.12 --offline --no-index --find-links (Join-Path $here "wheels") -c (Join-Path $here "{constraints}")
& $uv tool update-shell
Write-Host "Vexel installed. Open a new PowerShell window and run: vexel gui"
"""

INSTALL_SH = """#!/bin/sh
# Vexel offline installation (macOS / Linux). Run inside this folder:  sh install.sh
set -e
here="$(cd "$(dirname "$0")" && pwd)"
export UV_OFFLINE=1
export UV_PYTHON_INSTALL_MIRROR="file://$here/python"
"$here/uv/uv" python install 3.12 --no-bin
"$here/uv/uv" tool install "$here/{wheel}[agent,sem]" --python 3.12 --offline --no-index --find-links "$here/wheels" -c "$here/{constraints}"
"$here/uv/uv" tool update-shell || true
echo "Vexel installed. Open a new terminal and run: vexel gui"
"""


def run(*cmd: str, env: dict[str, str] | None = None) -> str:
    import os

    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT, env={**os.environ, **(env or {})})
    if proc.returncode:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{proc.stderr[-3000:]}")
    return proc.stdout


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vexel-offline-bundle"})
    with urllib.request.urlopen(request, timeout=300) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def python_url(key: str) -> str:
    out = run("uv", "python", "list", "3.12", "--all-platforms", "--show-urls", "--only-downloads")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("cpython-3.12.") and parts[0].endswith(key) and "freethreaded" not in parts[0]:
            return parts[1]
    raise SystemExit(f"no Python 3.12 download for {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "offline")
    args = parser.parse_args()
    spec = PLATFORMS[args.platform]
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    name = f"vexel-{version}-offline-{args.platform}"
    bundle = args.out / name
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "wheels").mkdir(parents=True)

    # Vexel and the locked versions
    run("uv", "lock", "--check")
    run("uv", "build", "--wheel", "--out-dir", str(bundle))
    wheel = next(bundle.glob("vexel-*.whl")).name
    constraints = f"vexel-{version}-constraints.txt"
    run("uv", "export", "--format", "requirements-txt", "--no-hashes", "--no-emit-project", "--no-dev", "--extra", "agent", "--extra", "sem",
        "--output-file", str(bundle / constraints))  # fmt: skip

    # the exact set of packages this platform needs, at those versions, as wheels
    reqs = bundle / "requirements-platform.txt"
    run("uv", "pip", "compile", "pyproject.toml", "--extra", "agent", "--extra", "sem", "--python-platform", spec["triple"], "--python-version", "3.12",
        "-c", str(bundle / constraints), "--no-header", "--no-annotate", "-o", str(reqs), env=spec.get("env"))  # fmt: skip
    tags = [x for t in spec["tags"] for x in ("--platform", t)]
    run("uv", "run", "--no-project", "--with", "pip", "python", "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
        *tags, "--python-version", "3.12", "--implementation", "cp", "-r", str(reqs), "-d", str(bundle / "wheels"))  # fmt: skip

    # uv itself, and the Python build it installs
    uv_version = run("uv", "--version").split()[1]
    archive = bundle / spec["uv_asset"]
    download(f"https://github.com/astral-sh/uv/releases/download/{uv_version}/{spec['uv_asset']}", archive)
    (bundle / "uv").mkdir()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(bundle / "uv")
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.isfile() and Path(member.name).name in ("uv", "uvx"):
                    member.name = Path(member.name).name
                    tf.extract(member, bundle / "uv", filter="data")
    archive.unlink()
    url = python_url(spec["python_key"])
    release, filename = url.split("/")[-2], url.split("/")[-1]
    download(url, bundle / "python" / release / urllib.request.url2pathname(filename))

    # install scripts, checksums, readme
    (bundle / "install.ps1").write_text(INSTALL_PS1.format(wheel=wheel, constraints=constraints), encoding="utf-8")
    (bundle / "install.sh").write_text(INSTALL_SH.format(wheel=wheel, constraints=constraints), encoding="utf-8", newline="\n")
    (bundle / "README.txt").write_text(
        f"Vexel {version} offline installation for {args.platform}. No network access is needed.\n\n"
        "Windows:        powershell -ExecutionPolicy Bypass -File .\\install.ps1\n"
        "macOS / Linux:  sh install.sh\n\n"
        "Then open a new terminal and run: vexel gui\n"
        "Check the files first (optional): SHA256SUMS.txt lists the SHA-256 of every file.\n",
        encoding="utf-8",
    )  # fmt: skip
    sums = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    zip_path = shutil.make_archive(str(args.out / name), "zip", root_dir=args.out, base_dir=name)
    size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()) / 1e6
    print(json.dumps({"bundle": str(bundle), "zip": zip_path, "files": len(sums), "size_mb": round(size)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
