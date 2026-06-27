#!/usr/bin/env bash
# Rebuild the data-free viewer app shell bundled with the Python pipeline.
#
# Run this whenever image_space/src/ (the React/PixiJS viewer) changes so that
# `imagespace` ships an up-to-date static site without requiring users to
# install Node/Vite. The shell (index.html + assets/ + .nojekyll + favicon.svg)
# is committed to git inside the imagespace_viewer_shell package; the pipeline
# copies it into every output directory.
#
# Usage:
#   bash scripts/build_viewer_shell.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_DIR="$REPO_ROOT/image_space"
SHELL_DIR="$REPO_ROOT/imagespace_viewer_shell/viewer_shell"
SRC_OUT="$VIEWER_DIR/output"

echo "Building viewer in $VIEWER_DIR ..."
cd "$VIEWER_DIR"
npx vite build

echo "Staging data-free shell into $SHELL_DIR ..."
rm -rf "$SHELL_DIR"
mkdir -p "$SHELL_DIR"
cp "$SRC_OUT/index.html" "$SHELL_DIR/index.html"
cp "$SRC_OUT/.nojekyll" "$SHELL_DIR/.nojekyll"
cp -r "$SRC_OUT/assets" "$SHELL_DIR/assets"

# Ship the ImageSpace logo as the favicon and point index.html at it (relative).
cp "$REPO_ROOT/assets/logo.svg" "$SHELL_DIR/favicon.svg"
# Replace the Vite default favicon reference with our relative favicon.
SHELL_DIR="$SHELL_DIR" python3 - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["SHELL_DIR"], "index.html")
s = p.read_text()
s = re.sub(r'<link rel="icon"[^>]*>', '<link rel="icon" type="image/svg+xml" href="./favicon.svg" />', s)
p.write_text(s)
PY

echo
echo "Shell staged. Contents:"
ls -la "$SHELL_DIR"
ls -la "$SHELL_DIR/assets"
echo
echo "Done. Commit imagespace_viewer_shell/viewer_shell/ to update the bundled viewer."