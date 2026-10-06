# Low-Re Tailfin Optimization

My NTU project on tailfin (vertical stabilizer) planform optimization for low
Reynolds number flight (Re ~ 50k–100k). The idea is a closed loop: parametric
CAD → OpenFOAM CFD with transition modelling → ML surrogate → wind tunnel
validation. This repo holds everything for the first three stages; the tunnel
campaign is next.

## What I found so far

- 30-case Latin Hypercube sweep (sweep 15–35°, AR 1.5–3.0, taper 0.4–0.8),
  all converged with the kOmegaSSTLM transition model.
- Drag surrogate (Gaussian Process) hits R² 0.991, so the optimum it predicts
  is trustworthy — and the confirmation run backed that up (predicted
  Cd 0.00970, measured 0.00998).
- Predicted optimum: **sweep ≈ 16.0°, AR ≈ 1.54, taper ≈ 0.42**
  (`tailfin_optimized.stl` at repo root, watertight and print-ready).
  Roughly: less sweep and area = less drag, which is what you'd expect.
- y+ honesty: I get wall-resolved means over the LSB region (mean ~0.7,
  90% of faces ≤ 1) but the stagnation strip and facet ridges spike higher.
  Forcing global max y+ < 1 broke every mesher setting I tried (documented in
  the commit history), so I report regional stats with explicit exclusions —
  the standard practice for LSB work. The tunnel data will calibrate the rest.

## Repo layout

- `cad_generator/` — parametric SD8020 tailfin builder (CadQuery).
- `openfoam_template/` — frozen case template: 30-point LHS matrix
  (`doe_matrix.csv`), kOmegaSSTLM setup, tuned snappyHexMesh dict.
- `runs/` — one dir per case (meshes/time dirs stay local, not committed).
- `run_doe.py` — batch orchestrator: clones the template per DoE row, meshes,
  solves (early-stops on residual + force convergence), extracts regional y+,
  writes `doe_results_summary.csv`. Survives per-case failures.
- `ml_optimizer/train_surrogate.py` — GPR/GBR/polynomial-RSM comparison with
  CV R²/RMSE/MAPE, parity + contour plots, global optimum search.
- `check_regional_yplus.py` — per-face y+ mapped to x/c with LSB-region stats.
- `gmsh_generator/` — experimental explicit-BL meshing route (parked; the
  Gmsh build I have only does 2D boundary layers).

## Running it

Requirements: macOS, OpenFOAM-v2606 app, Python 3 with `cadquery`,
`trimesh`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`.

Important: run everything from a **normal terminal**, not inside the
`openfoam` shell — the container Python doesn't have the CAD/ML packages
(`openfoam -c ...` is called by the scripts themselves).

```bash
cd ~/low_re_tailfin_project

# smoke test (mesh only, ~5 min)
python3 run_doe.py --cases 1-2 --no-solve

# full sweep (~15 min/case, ~7 h for 30)
nohup python3 run_doe.py > doe_batch.log 2>&1 &
tail -f doe_batch.log

# surrogate + optimum
python3 ml_optimizer/train_surrogate.py
```

Results land in `doe_results_summary.csv` (one row per case) with per-face
datasets under `results/`.

## Still to do

1. Print the baseline + optimized fins and run the tunnel campaign
   (force balance over β = 0–15°, oil-flow for the LSB footprint).
2. A sideslip CFD campaign for true side-force data, plus a proper LSB-length
   extraction from wall shear — that upgrades the current Cd-proxy optimum to
   the real objective.
3. Write it all up.
