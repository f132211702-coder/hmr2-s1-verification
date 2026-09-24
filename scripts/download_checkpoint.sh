#!/usr/bin/env bash
# Download the official pretrained HMR2.0 checkpoint + SMPL support data
# (source: https://github.com/shubham-goel/4D-Humans, published at the URL
# below by the paper authors) into ~/.cache/4DHumans/, which is where every
# script in this repo expects to find it.
#
# This replaces the auto-download logic buried inside the Python package
# (src/hmr2/models/__init__.py: download_models()) with a plain, inspectable
# shell command, so it's obvious what gets fetched from where.
#
# Usage:
#   bash scripts/download_checkpoint.sh
set -euo pipefail

CACHE_DIR="${HMR2_CACHE_DIR:-$HOME/.cache/4DHumans}"
URL="https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
ARCHIVE="/tmp/hmr2_data.tar.gz"

mkdir -p "$CACHE_DIR"

if [ -f "$CACHE_DIR/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt" ]; then
    echo "Checkpoint already present at $CACHE_DIR, nothing to do."
    exit 0
fi

echo "Downloading HMR2.0 checkpoint + SMPL support data from:"
echo "  $URL"
curl -L --fail "$URL" -o "$ARCHIVE"

echo "Extracting into $CACHE_DIR ..."
tar -xzf "$ARCHIVE" -C "$CACHE_DIR"
rm -f "$ARCHIVE"

echo "Done. Checkpoint is at:"
echo "  $CACHE_DIR/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
echo ""
echo "You still need the SMPL neutral body model separately (license"
echo "restriction, cannot be redistributed here): register and download"
echo "from https://smplify.is.tue.mpg.de/ and place the .pkl file at"
echo "$CACHE_DIR/data/smpl/SMPL_NEUTRAL.pkl"
