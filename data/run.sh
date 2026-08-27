#!/bin/bash

set -e

echo "==========================================="
echo "   Digit3D Dataset Generation Pipeline     "
echo "==========================================="

echo "[1/2] Constructing 3D Meshes from MNIST..."
# This generates the .obj files into src/train and src/test
python construct.py --spherical_z --post_process --decimate --target_triangles 500

# echo "[2/3] Precomputing Sparse Voxel SDFs..."
# This processes .obj files and creates .npz files into sdf/train and sdf/test
# python compute.py

echo "[2/3] Rendering Multi-View Normal and Depth Maps (64x64) with nvdiffrast..."
# This renders the 21-view Normal and Depth maps into mv_64/train and mv_64/test
python render_mv.py --split all --resolution 64 --output_folder mv_64 --front_delta 30.0

echo "[3/3] Rendering Multi-View Normal and Depth Maps (128x128) with nvdiffrast..."
# This renders the 21-view Normal and Depth maps into mv_128/train and mv_128/test
python render_mv.py --split all --resolution 128 --output_folder mv_128 --front_delta 30.0

echo "==========================================="
echo "Pipeline Completed Successfully!"
