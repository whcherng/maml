#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Install learn2learn on Python 3.12+
#
# The PyPI release of learn2learn ships without pre-generated .c
# files for its Cython extensions, and the stale ones reference
# `longintrepr.h` (removed in CPython 3.12). This script clones
# the repo, regenerates the .c files with a modern Cython, and
# installs from the local source.
# ──────────────────────────────────────────────────────────────────
set -e

echo "1/4  Installing Cython (build dependency)..."
pip install "Cython>=3.0.0"

echo "2/4  Cloning learn2learn..."
rm -rf /tmp/learn2learn
git clone --depth 1 https://github.com/learnables/learn2learn.git /tmp/learn2learn

echo "3/4  Regenerating Cython extensions for Python 3.12+..."
for pyx in /tmp/learn2learn/learn2learn/data/*.pyx; do
    echo "     cythonizing $(basename "$pyx")"
    cython "$pyx"
done

echo "4/4  Installing learn2learn..."
pip install /tmp/learn2learn

echo ""
echo "Done. Verifying import..."
python -c "import learn2learn; print(f'learn2learn {learn2learn.__version__} installed successfully')"
