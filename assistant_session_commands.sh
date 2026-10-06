#!/bin/zsh
# Every bash command executed by the assistant this session, in order.
# Workdir was /Users/yifanwang/low_re_tailfin_project unless noted.

# --- 1. progress audit ---
ls runs/case_01/postProcessing/forceCoeffs/
for i in $(seq -w 1 30); do d="runs/case_$i/postProcessing/forceCoeffs"; if [ -d "$d" ]; then n=$(ls "$d" | wc -l); echo "case_$i: has forceCoeffs ($n subdirs)"; else echo "case_$i: NO forceCoeffs"; fi; done
ls results/; ls ml_optimizer/
ls openfoam_template/system/ openfoam_template/constant/ openfoam_template/0/
cat openfoam_template/system/controlDict | head -60
cat openfoam_template/system/fvSolution | head -40
cat openfoam_template/constant/turbulenceProperties
tail -5 runs/case_01/postProcessing/forceCoeffs/0/coefficient.dat
ls runs/case_01/ | head -20
grep -h "End\|Final\|ExecutionTime" runs/case_01/log.simpleFoam | tail -5
ls runs/case_05/ runs/case_02/
ls runs/case_01/log.*
ls runs/case_*/log.simpleFoam
ls openfoam_template/postProcessing/forceCoeffs/
grep -c "tailfin" openfoam_template/system/controlDict
grep "patches" openfoam_template/system/controlDict
grep -A2 "geometry\|stl" openfoam_template/system/snappyHexMeshDict | head -20

# --- 2. y+ diagnosis ---
cat runs/case_01/postProcessing/yPlus/0/yPlus.dat | head -10
ls runs/case_01/postProcessing/yPlus/0/
cat openfoam_template/0/nut | head -40
cat openfoam_template/0/U | head -40
cat openfoam_template/0/k | head -30
cat openfoam_template/0/omega | head -40
cat runs/case_01/system/snappyHexMeshDict | tail -90
grep -i "layer\|Added\|Shrink\|Extrusion" runs/case_01/log.snappyHexMesh | tail -60
cat runs/case_01/constant/polyMesh/boundary | head -80

# --- 3. fixes applied (files edited, not bash) ---
# openfoam_template/system/snappyHexMeshDict : maxGlobalCells 2M->600k, level (4 5)->(3 4)->(4 5)->(3 4)->(4 5), features 3->2,
#   firstLayer 3e-06 -> 3.5e-05 -> 1.5e-05 -> 5e-07 -> 1e-06 -> 2e-06, layers 18->12->15->30->25->20,
#   expansion 1.12->1.2->1.25->1.3->1.2, featureAngle 60->30->60, nGrow 0->1->0, minThickness tightened,
#   maxThicknessToMedialRatio 0.3->0.5, minMedialAxisAngle 90->60, +slipFeatureAngle 30, locationInMesh -0.5->-0.2
# openfoam_template/0/nut : nutkWallFunction -> nutUSpaldingWallFunction
# openfoam_template/system/controlDict : endTime 1000 -> 500
# runs/case_01/system/controlDict : endTime 1 -> 500, startFrom startTime -> latestTime
# cad_generator/build_tailfin.py : export tolerance 0.1/0.1 -> 0.0001/0.05, +0.1% chord TE blunt
# run_batch.py : purge loop now keeps 0/ (tdir != "0")

# --- 4. recovery / rebuild ---
cp openfoam_template/0/p openfoam_template/0/U openfoam_template/0/k openfoam_template/0/omega openfoam_template/0/nut openfoam_template/0/gammaInt openfoam_template/0/ReThetat runs/case_01/0/
rm -rf runs/case_01/processor0 runs/case_01/processor1 runs/case_01/processor2 runs/case_01/processor3 runs/case_01/0/polyMesh
cp openfoam_template/system/snappyHexMeshDict runs/case_01/system/snappyHexMeshDict
cp openfoam_template/system/blockMeshDict runs/case_01/system/blockMeshDict
cp openfoam_template/0/nut runs/case_01/0/nut
ls -lh openfoam_template/constant/triSurface/ runs/case_01/constant/triSurface/
python3 -c 'import numpy as np; np.random.seed(42); n=30; sweeps=np.linspace(10.0,35.0,n); ars=np.linspace(1.5,3.5,n); tapers=np.linspace(0.4,0.8,n); np.random.shuffle(ars); np.random.shuffle(tapers); [print(f"case_{i+1:02d}: sweep={s:.2f} ar={a:.3f} taper={t:.3f}") for i,(s,a,t) in enumerate(zip(sweeps,ars,tapers))]'
python3 ~/low_re_tailfin_project/cad_generator/build_tailfin.py --sweep 10.0 --ar 3.362 --taper 0.400 --out ~/low_re_tailfin_project/runs/case_01/constant/triSurface/tailfin.stl
python3 -c 'import trimesh; m=trimesh.load("constant/triSurface/tailfin.stl", force="mesh"); print("faces:", len(m.faces))'
python3 -c 'import cadquery as cq; help(cq.exporters.export)' | head -40
head -5 ~/low_re_tailfin_project/cad_generator/sd8020.dat; tail -5 ~/low_re_tailfin_project/cad_generator/sd8020.dat

# --- 5. mesh + solve + y+ (workdir runs/case_01, via openfoam wrapper) ---
openfoam -c 'blockMesh && surfaceFeatureExtract && snappyHexMesh -overwrite'   # several iterations; later with log capture:
openfoam -c 'blockMesh >/dev/null 2>&1 && surfaceFeatureExtract >/dev/null 2>&1 && snappyHexMesh -overwrite 2>&1 | grep -E "Layer mesh|layers added|Finished meshing|FATAL|ERROR"'
openfoam -c 'simpleFoam 2>&1 | tail -25'
openfoam -c 'simpleFoam -postProcess -func yPlus -latestTime 2>&1 | tail -10'
openfoam -c 'nohup simpleFoam > log.simpleFoam 2>&1 &'
tail -15 log.simpleFoam
tail -3 log.simpleFoam
tail -8 log.simpleFoam
grep -c "^Time" log.simpleFoam
ps aux | grep -i simpleFoam | grep -v grep

# --- 6. regional y+ extraction ---
openfoam -c 'foamToVTK -latestTime -patch tailfin -fields "(yPlus wallShearStress p U)" 2>&1 | tail -8'   # failed: -patch invalid
openfoam -c 'foamToVTK -help 2>&1 | grep -i -A2 patch' | head -20
openfoam -c 'foamToVTK -latestTime -patches "(tailfin)" -fields "(yPlus wallShearStress)" 2>&1 | tail -6'  # ok -> VTK/case_01_300/...
ls VTK/
openfoam -c 'foamToVTK -help 2>&1 | grep -i -B1 -A1 ascii' | head -10
python3 check_regional_yplus.py --case runs/case_01
ls -la VTK/case_01_300/boundary/
head -40 VTK/case_01_300/boundary/tailfin.vtp
python3 -c 'import pandas as pd; d=pd.read_csv("postProcessing/forceCoeffs/0/coefficient_0.dat", comment="#", sep=r"\s+", header=None); print(d.tail(5).to_string(header=False))'
ls -la postProcessing/forceCoeffs/0/; ls -lat 100/ | head -8; ls -lat 300/ | head -8
grep -B2 -A6 "tailfin" constant/polyMesh/boundary | head -20
head -50 300/yPlus | head -60; grep -n "tailfin" 300/yPlus | head -5

# --- 7. history export (this request) ---
cp ~/.zsh_history ~/low_re_tailfin_project/zsh_history_raw.txt
python3 -c 'strip ": timestamp:elapsed;" prefix -> terminal_all_bash.txt'
