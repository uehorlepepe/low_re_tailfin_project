#!/usr/bin/env python3
"""Multi-objective planform optimization (Stage 3b): Pareto front Cd vs LSB.

Loads the trained winner surrogates (Cd, lsb_mean) and traces the tradeoff
with NSGA-II (pymoo; weighted-sum + scipy fallback if pymoo is missing) over
the proposal bounds [sweep 15-35, AR 1.5-3.0, taper 0.4-0.8].

Points outside the observed training hull are flagged extrapolated=True.
Saves ml_optimizer/pareto_optimal_designs.csv + pareto_front.png/pdf.

Usage: python3 ml_optimizer/optimize_pareto.py [--n-gen 100] [--pop 100]
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'ml_optimizer')
BOUNDS = np.array([[15.0, 35.0], [1.5, 3.0], [0.4, 0.8]])  # proposal space
NAMES = ['sweep_deg', 'aspect_ratio', 'taper_ratio']


def load_models():
    return (joblib.load(os.path.join(OUT, 'model_Cd.joblib')),
            joblib.load(os.path.join(OUT, 'model_lsb_mean.joblib')))


def observed_hull():
    d = pd.read_csv(os.path.join(ROOT, 'doe_results_summary.csv'))
    X = d[NAMES].to_numpy(float)
    return np.array([[X[:, i].min(), X[:, i].max()] for i in range(3)])


def predict(cd_m, lsb_m, X):
    return cd_m.predict(X), lsb_m.predict(X)


def pareto_nsga(cd_m, lsb_m, pop=100, n_gen=100, seed=7):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class Tailfin(Problem):
        def __init__(self):
            super().__init__(n_var=3, n_obj=2, xl=BOUNDS[:, 0], xu=BOUNDS[:, 1])

        def _evaluate(self, X, out, *a, **kw):
            c, l = predict(cd_m, lsb_m, X)
            out['F'] = np.column_stack([c, l])

    res = minimize(Tailfin(), NSGA2(pop_size=pop, seed=seed),
                   ('n_gen', n_gen), seed=seed, verbose=False)
    return res.X, res.F


def pareto_weighted(cd_m, lsb_m, n=51, seed=7):
    from scipy.optimize import differential_evolution
    rng = np.random.default_rng(seed)
    pts, vals = [], []
    # normalize with training ranges so weights are comparable
    d = pd.read_csv(os.path.join(ROOT, 'doe_results_summary.csv'))
    c0, c1 = d['Cd'].min(), d['Cd'].max()
    l0, l1 = d['lsb_mean'].min(), d['lsb_mean'].max()
    for w in np.linspace(0, 1, n):
        def f(x):
            c, l = predict(cd_m, lsb_m, np.atleast_2d(x))
            return w * (c[0] - c0) / (c1 - c0) + (1 - w) * (l[0] - l0) / (l1 - l0)
        r = differential_evolution(f, list(map(tuple, BOUNDS)), seed=seed,
                                   maxiter=60, polish=True)
        c, l = predict(cd_m, lsb_m, np.atleast_2d(r.x))
        pts.append(r.x)
        vals.append([c[0], l[0]])
    return np.array(pts), np.array(vals)


def nondominated(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            keep = keep & ~((F[:, 0] <= F[i, 0]) & (F[:, 1] <= F[i, 1])
                            & ((F[:, 0] < F[i, 0]) | (F[:, 1] < F[i, 1])))
            keep[i] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-gen', type=int, default=100)
    ap.add_argument('--pop', type=int, default=100)
    a = ap.parse_args()

    cd_m, lsb_m = load_models()
    hull = observed_hull()
    try:
        X, F = pareto_nsga(cd_m, lsb_m, pop=a.pop, n_gen=a.n_gen)
        method = f'nsga2 pop={a.pop} gen={a.n_gen}'
    except ImportError:
        X, F = pareto_weighted(cd_m, lsb_m)
        method = 'weighted-sum fallback'
    keep = nondominated(F)
    X, F = X[keep], F[keep]
    order = np.argsort(F[:, 0])
    X, F = X[order], F[order]
    extra = ((X < hull[:, 0]) | (X > hull[:, 1])).any(axis=1)

    df = pd.DataFrame(X, columns=NAMES)
    df['pred_Cd'] = F[:, 0]
    df['pred_lsb_mean'] = F[:, 1]
    df['extrapolated'] = extra
    csv = os.path.join(OUT, 'pareto_optimal_designs.csv')
    df.to_csv(csv, index=False)
    print(f'{method}: {len(df)} Pareto points -> {csv} '
          f'({int(extra.sum())} extrapolated)')

    # knee = max distance to chord between endpoints
    p0, p1 = F[0], F[-1]
    chord = p1 - p0
    dist = np.abs((F - p0)[:, 0] * chord[1] - (F - p0)[:, 1] * chord[0])
    knee = int(np.argmax(dist / (np.hypot(*chord) + 1e-12)))
    print(f'knee idx={knee} Cd={F[knee,0]:.5f} lsb={F[knee,1]:.3f} '
          f'sweep={X[knee,0]:.2f} AR={X[knee,1]:.3f} taper={X[knee,2]:.3f}')

    d = pd.read_csv(os.path.join(ROOT, 'doe_results_summary.csv'))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(d['Cd'], d['lsb_mean'], s=26, alpha=0.45, label='DoE (measured)')
    ax.plot(F[:, 0], F[:, 1], 'r-', lw=2, label='Pareto front (surrogate)')
    ax.scatter(F[:, 0], F[:, 1], s=22, c='r', alpha=0.7)
    ax.scatter([F[0, 0], F[knee, 0], F[-1, 0]],
               [F[0, 1], F[knee, 1], F[-1, 1]],
               s=90, facecolors='none', edgecolors='k', lw=1.5,
               label='min-Cd / knee / min-LSB')
    ax.set_xlabel('Drag coefficient Cd (predicted)')
    ax.set_ylabel('LSB proxy mean y+ (predicted)')
    ax.set_title('Tailfin planform Pareto front: drag vs LSB behavior')
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'pareto_front.{ext}'), dpi=150)
    plt.close(fig)
    print('saved pareto_front.png/pdf')


if __name__ == '__main__':
    main()
