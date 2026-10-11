# Low-Re Tailfin Optimization

My project on tailfin (vertical stabilizer) planform optimization for low
Reynolds number flight (Re around 50k to 100k). Closed loop: parametric CAD
to OpenFOAM CFD with transition modelling to ML surrogate to wind tunnel
validation. The first three stages are done and committed here. The headline
so far: my optimized fin buys **40% more directional stability for 14% more
drag** versus an RQ-11 Raven style baseline, with derivatives to prove it.

## Headline numbers

| what | result |
|---|---|
| DoE sweep | 30/30 converged (LHS: sweep 15-35 deg, AR 1.5-3.0, taper 0.4-0.8) |
| Drag surrogate (Gaussian Process) | R2 0.991, confirmed at 2.9% error |
| Predicted optimum | sweep 16.0 deg, AR 1.54, taper 0.42 (`tailfin_optimized.stl`, print-ready) |
| vs Raven NACA 0009 (Re-matched) | +43% side force per deg, +40% stiffer yaw, +14% drag |
| Sideslip pilot | 9/9 converged, CY slope linear and symmetric |
| Delta branch (45-60 deg sweep) | 20/20 converged, no tight LEV cores found |

## What I learned (honestly)

Small, lightly swept, low taper fins win on drag because wetted area
dominates everything at this scale. The bubble proxy resisted every model I
tried, and the Sobol split explains why: drag is an area story, bubbles are
a taper story. Three full campaigns (axial, sideslip to ±20 deg, 60 deg
deltas) all stayed attached with no mean-flow reversal, which scopes exactly
what the tunnel must test instead of guessing at it.

On mesh honesty: I hold wall-resolved means over the LSB zone (mean about
0.7, 90% of faces at or under 1) with explicit stagnation/ridge exclusions.
Forcing global max y+ under 1 broke every mesher setting on a laptop, and
the commit history shows the receipts.

## Repo layout

- `cad_generator/` : parametric SD8020/NACA 4-digit fin builder (CadQuery).
- `openfoam_template/` : frozen cruise template, 30-point LHS matrix,
  kOmegaSSTLM setup, tuned snappyHexMesh dict.
- `openfoam_template_hisweep/` : delta-branch template with tip/wake
  refinement boxes. `openfoam_template_pimple/` : staged transient setup
  (sanity-checked, parked for now).
- `runs/`, `runs_hisweep/`, `runs_aoa/` : one dir per case (meshes and time
  dirs stay local, never committed).
- `run_doe.py` : batch orchestrator. Clones the template per DoE row, meshes,
  solves (early-stops on residual plus force convergence), extracts regional
  y+, writes the summary CSV. Supports `--betas`, `--aoas`, and full campaign
  isolation flags. Survives per-case failures.
- `ml_optimizer/` : surrogate comparison (GPR/GBR/polynomial-RSM, R2/RMSE/
  MAPE, parity and contour plots), NSGA-II Pareto, Sobol indices, Cp/LSB/LEV
  extractors with publication plots.
- `check_regional_yplus.py` : per-face y+ mapped to chord fraction with
  LSB-region stats.

## Running it

Requirements: macOS, OpenFOAM-v2606 app, Python 3 with `cadquery`,
`trimesh`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`.

Important: run everything from a **normal terminal**, not inside the
`openfoam` shell. The container Python lacks the CAD/ML packages
(`openfoam -c ...` is called by the scripts themselves).

```bash
cd ~/low_re_tailfin_project

# smoke test (mesh only, about 5 min)
python3 run_doe.py --cases 1-2 --no-solve

# full sweep (about 15 min/case, about 7 h for 30)
nohup python3 run_doe.py > doe_batch.log 2>&1 &
tail -f doe_batch.log

# surrogate + optimum
python3 ml_optimizer/train_surrogate.py

# sideslip pilot on one geometry (about 2 h)
python3 run_doe.py --cases 25 --betas '-20,-15,-10,-5,0,5,10,15,20'
```

Results land in `doe_results_summary.csv` (one row per case) with per-face
datasets under `results/`.

## Still to do

1. Print the baseline plus optimized fins and run the tunnel campaign
   (force balance over beta 0 to 15 deg, oil-flow for the LSB footprint).
2. A beta-times-geometry campaign for per-planform stability derivatives.
3. Write it all up.
