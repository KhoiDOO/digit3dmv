#!/bin/bash

set -e

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/mv_64" ] || [ -d "$SCRIPT_DIR/src" ]; then
    DATA_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/data/mv_64" ] || [ -d "$SCRIPT_DIR/data/src" ]; then
    DATA_DIR="$SCRIPT_DIR/data"
else
    DATA_DIR="$SCRIPT_DIR"
fi

cd "$DATA_DIR"

echo "==========================================="
echo "   Digit3D Multi-View Dataset Archiving    "
echo "==========================================="
echo "Working directory: $(pwd)"

echo "[1/2] Compressing mv_64 -> mv_64.zip..."
if [ -d "mv_64" ]; then
    zip -r -q -1 mv_64.zip mv_64
    echo "  -> Created mv_64.zip ($(du -sh mv_64.zip | cut -f1))"
else
    echo "Warning: mv_64 directory not found in $(pwd)"
fi

echo "[2/2] Compressing mv_128 -> mv_128.zip..."
if [ -d "mv_128" ]; then
    zip -r -q -1 mv_128.zip mv_128
    echo "  -> Created mv_128.zip ($(du -sh mv_128.zip | cut -f1))"
else
    echo "Warning: mv_128 directory not found in $(pwd)"
fi

echo "==========================================="
echo "Archiving Completed Successfully!"
