#!/usr/bin/env python3
"""Global sensitivity analysis (Sobol indices) on the trained surrogates.

Quantifies how [sweep, AR, taper] control Cd and LSB behavior using
Saltelli sampling evaluated through the winner models in ml_optimizer/.

Bounds (proposal space): sweep 15-35 deg, AR 1.5-3.0, taper 0.4-0.8.
N=1024 base samples, second order -> 8192 surrogate evaluations per target.

Outputs: console S1/ST table, results/sobol_sensitivity_{cd,lsb}.csv,
ml_optimizer/sobol_indices.png/pdf.

Usage: python3 ml_optimizer/run_sobol.py [--n 1024] [--no-second-order]
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'ml_optimizer')
RES = os.path.join(ROOT, 'results')
NAMES = ['sweep_deg', 'aspect_ratio', 'taper_ratio']
PROBLEM = {'num_vars': 3, 'names': NAMES,
           'bounds': [[15.0, 35.0], [1.5, 3.0], [0.4, 0.8]]}
TARGETS = {'Cd': 'model_Cd.joblib', 'lsb_mean': 'model_lsb_mean.joblib'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--no-second-order', action='store_true')
    a = ap.parse_args()
    second = not a.no_second_order

    X = saltelli.sample(PROBLEM, a.n, calc_second_order=second)
    print(f'Saltelli: N={a.n} second_order={second} -> {len(X)} samples')
    os.makedirs(RES, exist_ok=True)

    frames = {}
    for target, model_file in TARGETS.items():
        mdl = joblib.load(os.path.join(OUT, model_file))
        y = mdl.predict(X)
        si = sobol.analyze(PROBLEM, y, calc_second_order=second,
                           print_to_console=False)
        df = pd.DataFrame({'parameter': NAMES, 'S1': si['S1'],
                           'S1_conf': si['S1_conf'], 'ST': si['ST'],
                           'ST_conf': si['ST_conf']})
        csv = os.path.join(RES, f'sobol_sensitivity_{target.split("_")[0].lower()}.csv')
        df.to_csv(csv, index=False)
        frames[target] = df
        print(f'--- {target} ---')
        print(df.to_string(index=False,
                           formatters={c: '{:.4f}'.format
                                       for c in ['S1', 'S1_conf', 'ST', 'ST_conf']}))
        print(f'saved {csv}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (target, df) in zip(axes, frames.items()):
        x = np.arange(len(df))
        ax.bar(x - 0.2, df['S1'], 0.4, yerr=df['S1_conf'], label='S1 first-order')
        ax.bar(x + 0.2, df['ST'], 0.4, yerr=df['ST_conf'], label='ST total-effect')
        ax.set_xticks(x)
        ax.set_xticklabels(['sweep', 'AR', 'taper'])
        ax.set_title(f'Sobol indices: {target}')
        ax.set_ylabel('sensitivity index')
        ax.legend(frameon=True, fontsize=9)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Global sensitivity of tailfin responses to planform parameters')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'sobol_indices.{ext}'), dpi=150)
    plt.close(fig)
    print('saved ml_optimizer/sobol_indices.png/pdf')


if __name__ == '__main__':
    main()
