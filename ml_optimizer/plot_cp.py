#!/usr/bin/env python3
"""Publication plots: chordwise Cp comparison across Pareto-key designs.

Reads results/cp_<tag>.csv + cp_<tag>.json, draws smoothed Cp(x/c) with
separation (o) / reattachment (^) markers. Saves cp_comparison.png/pdf.
Usage: python3 ml_optimizer/plot_cp.py optimum best-measured high-lsb
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'ml_optimizer')
LABELS = {'optimum': 'Optimum (16.0 deg / AR 1.54)',
          'best-measured': 'Best measured run_25 (16.4 deg / AR 1.65)',
          'high-lsb': 'High-LSB run_06 (33.5 deg / AR 1.85)'}


def main(tags):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tag in tags:
        d = np.loadtxt(os.path.join(RES, f'cp_{tag}.csv'), delimiter=',',
                       skiprows=1)
        m = json.load(open(os.path.join(RES, f'cp_{tag}.json')))
        o = np.argsort(d[:, 0])
        x, cs = d[o, 0], d[o, 2]
        lab = LABELS.get(tag, tag)
        ax.plot(x, cs, lw=1.8, label=f'{lab}  [LSB {m["x_sep"]:.2f}-{m["x_reattach"]:.2f}]')
        ax.plot(m['x_sep'], np.interp(m['x_sep'], x, cs), 'o', ms=7)
        rea_y = np.interp(m['x_reattach'], x, cs)
        ax.plot(m['x_reattach'], rea_y, '^', ms=8)
    ax.set_xlabel('Chord fraction x/c (mid-span)')
    ax.set_ylabel('Pressure coefficient Cp')
    ax.set_title('Chordwise Cp: separation plateau and reattachment')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'cp_comparison.{ext}'), dpi=150)
    plt.close(fig)
    print('saved cp_comparison.png/pdf')


if __name__ == '__main__':
    main(sys.argv[1:] or ['optimum', 'best-measured', 'high-lsb'])
