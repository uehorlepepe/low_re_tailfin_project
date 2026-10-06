#!/usr/bin/env python3
"""Chordwise Cp extraction + LSB plateau detection for a solved case.

Exports tailfin + inlet surface pressure via foamToVTK (ASCII), forms the
pressure coefficient Cp = (p - p_ref)/(0.5*U_inf^2) with p_ref = mean inlet
pressure (OpenFOAM incompressible p is kinematic, p/rho, so no rho needed),
maps faces to local x/c at the mid-span station, and locates the LSB from
the classic pressure plateau: suction peak -> flat dead-air region ->
recovery/reattachment.

Separation = first plateau face after the suction peak; reattachment = first
sustained-recovery face after the plateau (|dCp/dx| back above threshold).

Usage:
  python3 ml_optimizer/extract_cp.py --case runs/case_99 --sweep 15.97 \\
      --ar 1.54 --taper 0.417 --tag optimum
Outputs: results/cp_<tag>.csv (x/c, Cp, side) + results/cp_<tag>.json metrics.
U_inf default 6.6618 (template magUInf).
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from check_regional_yplus import parse_vtp_ascii  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_INF = 6.6618
Q = 0.5 * U_INF ** 2


def export_patch(case, patches, fields):
    cmd = ('foamToVTK -latestTime -no-internal -patches "(%s)" -fields "(%s)" '
           '-ascii -overwrite' % (' '.join(patches), ' '.join(fields)))
    p = subprocess.run(f'openfoam -c \'{cmd}\'', shell=True, cwd=case,
                       executable='/bin/zsh', capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f'foamToVTK failed: {(p.stderr or "")[-400:]}')


def find_vtp(case, patch):
    cands = []
    for dp, _, fn in os.walk(os.path.join(case, 'VTK')):
        for f in fn:
            if f.endswith('.vtp') and patch in f:
                cands.append(os.path.join(dp, f))
    if not cands:
        raise RuntimeError(f'no {patch}.vtp exported')
    return sorted(cands)[-1]


def station_cp(fc, cp, z_mid, half=0.05, span=1.0):
    """Faces near mid-span on one side (y>0 suction/pressure symmetric here)."""
    m = (np.abs(fc[:, 2] - z_mid) < half * span) & (fc[:, 1] > 0)
    return fc[m], cp[m]


def detect_lsb(xc, cp_s, grad_tol_frac=0.08, min_len=0.03, n_bins=200):
    """Plateau bookkeeping on binned+smoothed Cp sorted by x/c. Returns dict."""
    # bin to uniform x/c (kills duplicate-x NaNs from unstructured bands)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(xc, edges) - 1, 0, n_bins - 1)
    bx = np.full(n_bins, np.nan)
    bc = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bx[b] = xc[m].mean()
            bc[b] = cp_s[m].mean()
    ok = np.isfinite(bx) & np.isfinite(bc)
    x, c = bx[ok], bc[ok]
    w = max(3, len(x) // 40)
    c = np.convolve(c, np.ones(w) / w, mode='same')
    peak = int(np.argmin(c))
    grad = np.gradient(c, x)
    scale = max(np.abs(grad[peak:]).max(), 1e-9)
    tol = grad_tol_frac * scale
    flat = np.abs(grad) < tol
    i = peak
    while i < len(x) - 1 and not flat[i + 1]:
        i += 1
    sep = i + 1 if i + 1 < len(x) else peak
    j = sep
    while j < len(x) - 1 and flat[j + 1]:
        j += 1
    # reattachment: recovery sustained over 5 bins, searched before x=0.95
    k = j + 1
    while k < len(x) - 6 and x[k] < 0.95 and not (
            (grad[k:k + 5] > tol).all()):
        k += 1
    rea = min(k, len(x) - 1)
    length = float(x[rea] - x[sep]) if rea > sep else 0.0
    if length < min_len:
        rea, length = sep, 0.0
    return dict(x_sep=float(x[sep]), x_reattach=float(x[rea]),
                bubble_length=float(length), x_suction_peak=float(x[peak]),
                cp_min=float(c[peak]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--sweep', type=float, required=True)
    ap.add_argument('--ar', type=float, required=True)
    ap.add_argument('--taper', type=float, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--u-inf', type=float, default=U_INF)
    a = ap.parse_args()

    case = os.path.join(ROOT, a.case) if not os.path.isabs(a.case) else a.case
    export_patch(case, ['tailfin', 'inlet'], ['p'])
    fc_t, p_t = parse_vtp_ascii(find_vtp(case, 'tailfin'), field='p')
    _, p_in = parse_vtp_ascii(find_vtp(case, 'inlet'), field='p')
    p_ref = float(np.mean(p_in))
    q = 0.5 * a.u_inf ** 2
    cp = (p_t - p_ref) / q

    root, tip = 0.15, 0.15 * a.taper
    span = a.ar * 0.5 * (root + tip)
    off = span * math.tan(math.radians(a.sweep))
    z = np.clip(fc_t[:, 2], 0, span)
    chord = root + (tip - root) * z / span
    xc = (fc_t[:, 0] - off * z / span) / chord

    fcs, cps = station_cp(fc_t, cp, 0.5 * span, span=span)
    srt = np.argsort((fcs[:, 0] - off * 0.5) / (root + tip) / 2)
    xs = ((fcs[srt, 0] - off * 0.5) / (root + (tip - root) * 0.5))
    # smooth with moving average before detection
    w = max(5, len(xs) // 60)
    ker = np.ones(w) / w
    csm = np.convolve(cps[srt], ker, mode='same')
    lsb = detect_lsb(xs, csm)
    lsb.update(tag=a.tag, case=a.case, n_faces=int(len(fc_t)),
               p_ref=float(p_ref), cp_range=[float(cp.min()), float(cp.max())])

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    np.savetxt(os.path.join(ROOT, 'results', f'cp_{a.tag}.csv'),
               np.column_stack([xs, cps[srt], csm]),
               delimiter=',', header='x_over_c,Cp,Cp_smooth', comments='')
    with open(os.path.join(ROOT, 'results', f'cp_{a.tag}.json'), 'w') as f:
        json.dump(lsb, f, indent=2)
    print(f"[{a.tag}] x_sep={lsb['x_sep']:.3f} x_reattach={lsb['x_reattach']:.3f} "
          f"L_LSB={lsb['bubble_length']:.3f} Cp_min={lsb['cp_min']:.3f} "
          f"Cp_range=[{lsb['cp_range'][0]:.2f},{lsb['cp_range'][1]:.2f}]")


if __name__ == '__main__':
    main()
