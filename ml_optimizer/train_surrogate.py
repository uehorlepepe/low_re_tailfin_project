#!/usr/bin/env python3
"""Surrogate optimizer for the tailfin DoE (Stage 3).

Trains Gaussian-Process (Matern/Kriging), Gradient-Boosting, and quadratic
Polynomial-RSM surrogates on doe_results_summary.csv mapping
[sweep, AR, taper] -> [Cd, Cl, LSB mean, y+ avg], compares them by 5-fold CV
R^2 (plus out-of-fold RMSE/MAPE and LOO RMSE), keeps each target's winner,
and globally optimizes the planform for minimum predicted drag with an
LSB-bubble penalty proxy.

Objective honesty note: the proposal's J = CY/CD - a*(dLSB/c_root) needs
sideslip side-force and extracted LSB axial length, neither of which the
axial-flow sweep measured (Cy ~ 0, Cl ~ 0 symmetric). This script therefore
optimizes the measurable proxy J' = Cd + alpha * lsb_mean (alpha weights the
bubble proxy; default 0.005). Re-run with --alpha 0 for pure Cd, and revisit
once a sideslip campaign + wall-shear LSB-length extraction exist.

Outputs (ml_optimizer/): surrogate_report.txt, optimum_params.json,
model_<target>.joblib, parity_<target>.png.

Usage:
  python3 ml_optimizer/train_surrogate.py [--alpha 0.005] [--objective joint]
  objectives: joint | cd | lsb
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'doe_results_summary.csv')
OUT = os.path.join(ROOT, 'ml_optimizer')
FEATURES = ['sweep_deg', 'aspect_ratio', 'taper_ratio']


def load(path, ok_only=True):
    d = pd.read_csv(path)
    if ok_only:
        d = d[d['status'] == 'OK'].reset_index(drop=True)
    X = d[FEATURES].to_numpy(float)
    return d, X


def candidates(n=20000, seed=7, bounds=None):
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    span = bounds[:, 1] - bounds[:, 0]
    return lo + rng.random((n, 3)) * span


def mape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / np.where(y != 0, y, np.nan)))) * 100.0


def fit_and_score(X, y, seed=7):
    gpr = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=(ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern(nu=1.5, length_scale=np.ones(3),
                             length_scale_bounds=(1e-2, 1e3))
                    + WhiteKernel(1e-6, (1e-10, 1e-2))),
            n_restarts_optimizer=10, random_state=seed))
    gbr = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                 learning_rate=0.05, random_state=seed))
    prs = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1.0, random_state=seed))
    out = {}
    cv = min(5, len(X))
    for name, mdl in (('gpr', gpr), ('gbr', gbr), ('prs', prs)):
        r2 = cross_val_score(mdl, X, y, cv=cv, scoring='r2')
        oof = cross_val_predict(mdl, X, y, cv=cv)  # out-of-fold predictions
        rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
        loo = -cross_val_score(mdl, X, y, cv=LeaveOneOut(),
                               scoring='neg_root_mean_squared_error')
        mdl.fit(X, y)
        out[name] = dict(model=mdl, cv_r2=r2, oof_rmse=rmse,
                         oof_mape=mape(y, oof), loo_rmse=loo)
    return out


def contour_plot(model, target, bounds, fixed, path):
    """Sweep x AR response slice at median taper (winner model)."""
    n = 60
    sw = np.linspace(bounds[0, 0], bounds[0, 1], n)
    ar = np.linspace(bounds[1, 0], bounds[1, 1], n)
    SW, AR = np.meshgrid(sw, ar)
    G = np.column_stack([SW.ravel(), AR.ravel(),
                         np.full(SW.size, fixed)])
    Z = model.predict(G).reshape(SW.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(SW, AR, Z, levels=20)
    ax.contour(SW, AR, Z, levels=8, colors='k', linewidths=0.4, alpha=0.5)
    fig.colorbar(cs, ax=ax, label=f'predicted {target}')
    ax.set_xlabel('sweep_deg')
    ax.set_ylabel('aspect_ratio')
    ax.set_title(f'{target} response (taper={fixed:.3f})')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parity_plot(y, pred, std, target, path, r2):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=28, alpha=0.8)
    if std is not None:
        ax.errorbar(y, pred, yerr=1.96 * std, fmt='none', alpha=0.3)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel(f'measured {target}')
    ax.set_ylabel(f'predicted {target}')
    ax.set_title(f'{target} parity (5-fold CV R2={r2:.3f})')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.005)
    ap.add_argument('--objective', default='joint', choices=('joint', 'cd', 'lsb'))
    ap.add_argument('--data', default=None, help='input summary CSV')
    ap.add_argument('--outdir', default=None, help='artifact output dir')
    a = ap.parse_args()
    global OUT
    if a.outdir:
        OUT = os.path.abspath(a.outdir)
    os.makedirs(OUT, exist_ok=True)
    DATA_LOCAL = os.path.abspath(a.data) if a.data else DATA

    d, X = load(DATA_LOCAL)
    print(f'dataset: {len(d)} OK rows')
    bounds = np.array([[X[:, i].min(), X[:, i].max()] for i in range(3)])
    print('search bounds (observed):',
          dict(zip(FEATURES, [tuple(np.round(b, 3)) for b in bounds])))

    targets = {'Cd': d['Cd'].to_numpy(float),
               'Cl': d['Cl'].to_numpy(float),
               'lsb_mean': d['lsb_mean'].to_numpy(float),
               'yplus_avg': d['yplus_avg'].to_numpy(float)}

    report = [f'rows={len(d)} alpha={a.alpha} objective={a.objective}']
    models = {}
    for tname, y in targets.items():
        res = fit_and_score(X, y)
        win = max(res, key=lambda k: res[k]['cv_r2'].mean())
        w = res[win]
        models[tname] = w['model']
        joblib.dump(w['model'], os.path.join(OUT, f'model_{tname}.joblib'))
        scores = ' / '.join(
            f'{k} R2={res[k]["cv_r2"].mean():.3f} '
            f'RMSE={res[k]["oof_rmse"]:.5f} MAPE={res[k]["oof_mape"]:.2f}%'
            for k in ('gpr', 'gbr', 'prs'))
        line = (f'{tname}: winner={win} '
                f'CV_R2={w["cv_r2"].mean():.3f}+/-{w["cv_r2"].std():.3f} '
                f'OOF_RMSE={w["oof_rmse"]:.5f} OOF_MAPE={w["oof_mape"]:.2f}% '
                f'LOO_RMSE={w["loo_rmse"].mean():.5f} [{scores}]')
        print(line, flush=True)
        report.append(line)
        pred = w['model'].predict(X)
        std = None
        try:
            _, std = w['model'].steps[-1][1].predict(
                w['model'].steps[0][1].transform(X), return_std=True)
        except Exception:
            pass
        parity_plot(y, pred, std, tname,
                    os.path.join(OUT, f'parity_{tname}.png'),
                    w['cv_r2'].mean())
        contour_plot(w['model'], tname, bounds, float(np.median(X[:, 2])),
                     os.path.join(OUT, f'contour_{tname}.png'))

    # global optimum over dense random search on the joint proxy
    mdl_cd = models.get('Cd') if a.objective in ('joint', 'cd') else None
    mdl_lsb = models.get('lsb_mean') if a.objective in ('joint', 'lsb') else None
    grid = candidates(20000, bounds=bounds)
    j = np.zeros(len(grid))
    detail = {}
    if mdl_cd is not None:
        p = mdl_cd.predict(grid)
        j = j + p
        detail['pred_Cd'] = p
    if mdl_lsb is not None:
        p = mdl_lsb.predict(grid)
        j = j + a.alpha * p
        detail['pred_lsb'] = p
    best = grid[int(np.argmin(j))]
    opt = dict(sweep_deg=float(best[0]), aspect_ratio=float(best[1]),
               taper_ratio=float(best[2]), objective=a.objective,
               alpha=a.alpha, Jprime=float(j.min()))
    if 'pred_Cd' in detail:
        opt['pred_Cd'] = float(detail['pred_Cd'][np.argmin(j)])
    if 'pred_lsb' in detail:
        opt['pred_lsb_mean'] = float(detail['pred_lsb'][np.argmin(j)])
    # nearest measured row for grounding
    dist = np.abs((X - best) / (bounds[:, 1] - bounds[:, 0])).sum(axis=1)
    nr = d.iloc[int(np.argmin(dist))]
    opt['nearest_measured'] = {k: (nr[k].item() if hasattr(nr[k], 'item') else nr[k])
                               for k in ['run_id', 'case', 'Cd', 'lsb_mean']}
    with open(os.path.join(OUT, 'optimum_params.json'), 'w') as f:
        json.dump(opt, f, indent=2)
    print('OPTIMUM:', json.dumps(opt, indent=2))
    report.append('OPTIMUM: ' + json.dumps(opt))
    with open(os.path.join(OUT, 'surrogate_report.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    print('wrote ml_optimizer/{surrogate_report.txt, optimum_params.json, '
          'models, parity + contour PNGs}')


if __name__ == '__main__':
    main()
