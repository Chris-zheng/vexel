#!/usr/bin/env bash
# Build what testers install: the wheel, the exact dependency versions it was tested with
# (from uv.lock, for every platform), and SHA-256 checksums of both.
#
#   scripts/build_dist.sh            -> dist/vexel-<version>-py3-none-any.whl
#                                       dist/vexel-<version>-constraints.txt
#                                       dist/SHA256SUMS.txt
#
# Testers install with:  uv tool install ./vexel-<version>-py3-none-any.whl -c ./vexel-<version>-constraints.txt
set -euo pipefail
cd "$(dirname "$0")/.."

uv lock --check                       # uv.lock must match pyproject.toml
rm -rf dist
uv build --wheel --out-dir dist
version=$(uv run --no-sync python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
uv export --format requirements-txt --no-hashes --no-emit-project --no-dev --extra agent --output-file "dist/vexel-${version}-constraints.txt" >/dev/null
files=("vexel-${version}-py3-none-any.whl" "vexel-${version}-constraints.txt")
(cd dist && if command -v sha256sum >/dev/null; then sha256sum "${files[@]}"; else shasum -a 256 "${files[@]}"; fi) > dist/SHA256SUMS.txt
echo "built:"; ls -1 dist
