#!/usr/bin/env python3
"""True LSB length from saved wallShearStress fields (no new CFD).

For each solved case: foamToVTK ASCII export of the tailfin wallShearStress
vector, mid-span station, bin streamwise skin friction Cf_x over x/c, and
locate separation (Cf_x + -> -) and reattachment (Cf_x - -> +). This is the
proposal's Delta_x_LSB term, replacing the y+-mean proxy.

Usage:
  python3 ml_optimizer/extract_lsb.py [--cases 1-3] [--extra case_99:15.9678:1.54197:0.41746]
Output: results/lsb_lengths.csv + console table.
~1-2 min/case (VTK export dominates); full 30 take ~45 min - run in background.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from check_regional_yplus import parse_vtp_ascii  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, 'doe_results_summary.csv')
OUT = os.path.join(ROOT, 'results', 'lsb_lengths.csv')
N_BINS = 200


def vtp_of(case, patch):
    cands = []
    for dp, _, fn in os.walk(os.path.join(case, 'VTK')):
        for f in fn:
            if f.endswith('.vtp') and patch in f:
                cands.append(os.path.join(dp, f))
    if not cands:
        raise RuntimeError('no vtp exported')
    return sorted(cands)[-1]


def case_entry(case_dir, sweep, ar, taper, run_id, side='plus', beta=0.0):
    export = ('foamToVTK -latestTime -no-internal -patches "(tailfin)" '
              '-fields "(wallShearStress)" -ascii -overwrite')
    p = subprocess.run(f'openfoam -c \'{export}\'', shell=True, cwd=case_dir,
                       executable='/bin/zsh', capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f'foamToVTK failed: {(p.stderr or "")[-300:]}')
    try:
        fc, tau = parse_vtp_ascii(vtp_of(case_dir, 'tailfin'),
                                  field='wallShearStress', ncomp=3)
    finally:
        shutil.rmtree(os.path.join(case_dir, 'VTK'), ignore_errors=True)
    root, tip = 0.15, 0.15 * taper
    span = ar * 0.5 * (root + tip)
    off = span * math.tan(math.radians(sweep))
    z = np.clip(fc[:, 2], 0, span)
    chord = root + (tip - root) * z / span
    xc = (fc[:, 0] - off * z / span) / chord
    cfx = tau[:, 0]  # streamwise skin friction, U = +x
    # side selection: beta > 0 puts windward at -y (see write_inflow).
    # 'plus' preserves the legacy y>0-only behavior for beta=0 datasets.
    if side == 'both':
        smask = np.ones(len(fc), bool)
    else:
        sign = {'plus': 1.0, 'windward': -1.0 if beta >= 0 else 1.0,
                'leeward': 1.0 if beta >= 0 else -1.0}[side]
        smask = (fc[:, 1] * sign) > 0
    m = (np.abs(z - 0.5 * span) < 0.05 * span) & smask & np.isfinite(cfx)
    if m.sum() < 50:
        raise RuntimeError(f'too few station faces ({int(m.sum())})')

    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(xc[m], edges) - 1, 0, N_BINS - 1)
    bc = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        v = cfx[m][idx == b]
        if len(v):
            bc[b] = np.median(v)
    ok = np.isfinite(bc)
    xb = (edges[:-1] + edges[1:]) / 2
    x, c = xb[ok], bc[ok]
    # NOTE: OpenFOAM wallShearStress = stress ON the fluid; with U=+x the
    # attached layer reads NEGATIVE tau_x, reversed bubble flow POSITIVE.
    eps = 0.02 * max(np.abs(c).max(), 1e-12)
    pos = c > eps
    # separation: first sustained-positive bin past the LE (x > 0.02)
    sep = next((int(np.where(ok)[0][i]) for i in range(len(x))
                if x[i] > 0.02 and pos[i:i + 3].all()), None)
    if sep is None:
        return dict(run_id=run_id, x_sep='', x_reattach='', bubble_length=0.0,
                    status='attached')
    bi = list(x).index(x[sep]) if x[sep] in x else 0
    rea = next((i for i in range(bi + 1, len(x) - 4)
                if (c[i:i + 5] < -eps).all()), None)
    if rea is None:
        return dict(run_id=run_id, x_sep=round(float(x[bi]), 3),
                    x_reattach='', bubble_length='',
                    status='open-to-TE')
    return dict(run_id=run_id, x_sep=round(float(x[bi]), 3),
                x_reattach=round(float(x[rea]), 3),
                bubble_length=round(float(x[rea] - x[bi]), 3),
                status='closed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=None, help='"1-30", default all')
    ap.add_argument('--extra', default=None,
                    help='"case_99:15.9678:1.54197:0.41746" (repeatable)')
    ap.add_argument('--side', default='plus',
                    choices=('plus', 'windward', 'leeward', 'both'),
                    help='which surface to evaluate (needs --beta for '
                         'windward/leeward under sideslip)')
    ap.add_argument('--beta', type=float, default=0.0,
                    help='sideslip deg of the solved cases (sign convention '
                         'must match write_inflow)')
    a = ap.parse_args()
    d = pd.read_csv(SUMMARY)
    if a.cases:
        want = set()
        for part in a.cases.split(','):
            if '-' in part:
                x, y = part.split('-', 1)
                want.update(range(int(x), int(y) + 1))
            else:
                want.add(int(part))
        # exact case_XX match so beta suffixed dirs (case_25_b+010) are excluded
        d = d[d['case'].isin({f'case_{i:02d}' for i in want})]
    rows = []
    for _, r in d.iterrows():
        case_dir = os.path.join(ROOT, 'runs', r['case'])
        beta = float(r['beta_deg']) if 'beta_deg' in r and str(r['beta_deg']) != '' else a.beta
        try:
            out = case_entry(case_dir, float(r['sweep_deg']),
                             float(r['aspect_ratio']), float(r['taper_ratio']),
                             r['run_id'], side=a.side, beta=beta)
            out.update(Cd=float(r['Cd']), side=a.side, beta_deg=beta)
        except Exception as e:  # noqa: BLE001 - batch must survive
            out = dict(run_id=r['run_id'], x_sep='', x_reattach='',
                       bubble_length='', status=f'FAILED: {str(e)[-120:]}',
                       Cd=float(r['Cd']), side=a.side, beta_deg=beta)
        rows.append(out)
        print(f"[{out['run_id']}|{a.side}|b={beta}] {out['status']} "
              f"sep={out['x_sep']} rea={out['x_reattach']} L={out['bubble_length']}",
              flush=True)
    if a.extra:
        for spec in a.extra.split(';'):
            case, sw, ar, ta = spec.split(':')
            try:
                out = case_entry(os.path.join(ROOT, 'runs', case),
                                 float(sw), float(ar), float(ta), case,
                                 side=a.side, beta=a.beta)
                out.update(Cd='', side=a.side, beta_deg=a.beta)
            except Exception as e:  # noqa: BLE001
                out = dict(run_id=case, x_sep='', x_reattach='',
                           bubble_length='', status=f'FAILED: {str(e)[-120:]}',
                           Cd='', side=a.side, beta_deg=a.beta)
            rows.append(out)
            print(f"[{out['run_id']}|{a.side}|b={a.beta}] {out['status']} "
                  f"sep={out['x_sep']} rea={out['x_reattach']} L={out['bubble_length']}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'wrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
