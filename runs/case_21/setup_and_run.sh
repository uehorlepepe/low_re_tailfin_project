#!/bin/bash
set -e

echo "=========================================================="
echo "  LOCKING IN LOW-RE TAILFIN PARAMETERS (v2606)"
echo "  Target Re : 75,000 (Range 50,000 - 100,000)"
echo "  Velocity  : 6.6618 m/s (c = 0.17m, nu = 1.51e-5 m^2/s)"
echo "  Mesh y+   : <= 1.0 (firstLayerThickness = 3.0 microns)"
echo "=========================================================="

# 1. Update 0/U boundary condition and internal field
python3 -c '
import re, os
u_file = "0/U"
if os.path.exists(u_file):
    with open(u_file, "r") as f:
        txt = f.read()
    txt = re.sub(r"internalField\s+uniform\s+\([^)]+\);", "internalField   uniform (6.6618 0 0);", txt)
    txt = re.sub(r"value\s+uniform\s+\([^)]+\);", "value           uniform (6.6618 0 0);", txt)
    with open(u_file, "w") as f:
        f.write(txt)
    print("✓ Updated 0/U (U_inf = 6.6618 m/s)")
'

# 2. Update system/snappyHexMeshDict
python3 -c '
import re, os
path = "system/snappyHexMeshDict"
if os.path.exists(path):
    with open(path, "r") as f:
        txt = f.read()

    # Enable addLayers
    txt = re.sub(r"addLayers\s+false;", "addLayers       true;", txt)

    block = """addLayersControls
{
    relativeSizes         false;
    nBufferCellsNoExtrude 0;
    featureAngle          60;
    nGrow                 0;

    layers
    {
        "tailfin.*"
        {
            nSurfaceLayers 18;
        }
    }

    expansionRatio      1.12;
    firstLayerThickness 0.000003;
    minThickness        1e-07;

    nSolveIter          50;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nLayerIter          50;
    nRelaxedIter        20;
}"""
    txt = re.sub(r"addLayersControls[\s\S]*?\n\}", block, txt)
    with open(path, "w") as f:
        f.write(txt)
    print("✓ Updated system/snappyHexMeshDict (y1 = 3.0 um, 18 layers)")
'

# 3. Update system/controlDict with forceCoeffs & LSB function objects
python3 -c '
import re, os
path = "system/controlDict"
if os.path.exists(path):
    with open(path, "r") as f:
        txt = f.read()

    funcs = """
functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        timeInterval    1;
        log             true;
        
        patches         ("tailfin");
        rho             rhoInf;
        rhoInf          1.225;
        
        dragDir         (1 0 0);
        liftDir         (0 0 1);
        pitchAxis       (0 1 0);
        
        magUInf         6.6618;
        lRef            0.17;
        Aref            0.0255;
    }

    wallShearStress
    {
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         ("tailfin");
    }

    yPlus
    {
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         ("tailfin");
    }

    Q
    {
        type            Q;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }
}
"""
    if "functions" in txt:
        txt = re.sub(r"functions[\s\S]*$", funcs.strip(), txt)
    else:
        txt += "\n" + funcs.strip()

    with open(path, "w") as f:
        f.write(txt)
    print("✓ Updated system/controlDict (Force Coefficients, wallShearStress, yPlus, Q-criterion)")
'

echo "----------------------------------------------------------"
echo "Running snappyHexMesh..."
snappyHexMesh -overwrite | tee log.snappyHexMesh

echo "Checking mesh aspect ratio..."
checkMesh | grep -i "aspect"

echo "Cleaning previous solution time directories..."
rm -rf [1-9]* 1000

echo "Running simpleFoam (1000 iterations)..."
simpleFoam | tee log.simpleFoam

echo "----------------------------------------------------------"
echo "SIMULATION COMPLETE."
echo "Calculating final yPlus distribution..."
simpleFoam -postProcess -func yPlus -latestTime
echo "=========================================================="
