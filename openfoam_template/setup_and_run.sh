#!/bin/bash
set -e

# Reset domain mesh before snapping
blockMesh > log.blockMesh

# Clean up extended edge meshes from previous runs
rm -rf constant/extendedFeatureEdgeMesh

# Execute snappyHexMesh
snappyHexMesh -overwrite | tee log.snappyHexMesh

# Clean previous time directories
rm -rf [1-9]* 1000

# Run solver
simpleFoam | tee log.simpleFoam
