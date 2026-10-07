#!/usr/bin/env python3
"""LEV check on hisweep cases using saved Q-criterion fields (no new CFD).

For each OK row: write system/sampleDict with two cloud-sampled crossflow
planes (x = 0.35, 0.75; y in [-0.35, 0.35]; z in [0, 0.45]), run the OpenFOAM
`sample` utility on the latest time, and reduce to per-plane max Q plus area
fractions above absolute thresholds. Strong coherent cores => LEV regime.

Usage:
  python3 ml_optimizer/extract_lev.py [--matrix DOE] [--runs DIR] [--cases 1-20]
Output: results/lev_metrics.csv + results/lev_planes/<case>_p<plane>.csv
~1 min/case; full 20 in background.
"""
import argparse
import base64
import glob
import math
import os
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = (0.01, 0.05, 0.2)  # 1/s^2 absolute Q levels (low-Re scale)
PLANES = (0.35, 0.75)

DTYPES = {'Float32': ('<f4', 4), 'Float64': ('<f8', 8),
          'Int32': ('<i4', 4), 'Int64': ('<i8', 8),
          'UInt32': ('<u4', 4), 'UInt64': ('<u8', 8)}


def vtp_cells(path):
    """Cell centres + count from connectivity/offsets (any int width)."""
    tree = ET.parse(path)
    root = tree.getroot()
    conn = offs = None
    for e in root.iter('DataArray'):
        if e.attrib.get('Name') == 'connectivity':
            code, _ = DTYPES[e.attrib.get('type', 'Int32')]
            raw = base64.b64decode(''.join((e.text or '').split()))
            (nbytes,) = struct.unpack('<Q', raw[:8])
            conn = np.frombuffer(raw[8:8 + nbytes], dtype=np.dtype(code))
        if e.attrib.get('Name') == 'offsets':
            code, _ = DTYPES[e.attrib.get('type', 'Int32')]
            raw = base64.b64decode(''.join((e.text or '').split()))
            (nbytes,) = struct.unpack('<Q', raw[:8])
            offs = np.frombuffer(raw[8:8 + nbytes], dtype=np.dtype(code))
    pts = vtp_field(path, 'Points')
    starts = np.concatenate([[0], offs[:-1]])
    fc = np.array([pts[conn[s:e]].mean(axis=0) for s, e in zip(starts, offs)])
    return fc


def vtp_field(path, name):
    """Decode one inline-binary DataArray from a .vtp file."""
    tree = ET.parse(path)
    for e in tree.getroot().iter('DataArray'):
        if e.attrib.get('Name') == name:
            code, _ = DTYPES[e.attrib.get('type', 'Float32')]
            ncomp = int(e.attrib.get('NumberOfComponents', 1))
            raw = base64.b64decode(''.join((e.text or '').split()))
            (nbytes,) = struct.unpack('<Q', raw[:8])  # UInt64 header
            arr = np.frombuffer(raw[8:8 + nbytes], dtype=np.dtype(code))
            return arr.reshape(-1, ncomp) if ncomp > 1 else arr
    raise KeyError(f'{name} not in {path}')


def write_surfaces_dict(case_dir):
    planes = '\n'.join(
        f'    p{k}\n    {{\n        type cuttingPlane;\n'
        f'        planeType pointAndNormal;\n'
        f'        pointAndNormalDict {{ point ({xp:.3f} 0.0 0.2); normal (1 0 0); }}\n'
        f'    }}' for k, xp in enumerate(PLANES))
    txt = ('FoamFile { version 2.0; format ascii; class dictionary; '
           'object surfaces; }\ntype surfaces;\nlibs ("libsampling.so");\n'
           'writeControl writeTime;\nwriteInterval 1;\nsurfaceFormat vtk;\n'
           'interpolate true;\nfields ( Q );\n'
           f'surfaces\n(\n{planes}\n);\n')
    with open(os.path.join(case_dir, 'system', 'surfaces'), 'w') as f:
        f.write(txt)


def write_sample_dict(case_dir, time):
    ys = np.linspace(-0.35, 0.35, NY)
    zs = np.linspace(0.0, 0.45, NZ)
    sets = []
    for k, xp in enumerate(PLANES):
        pts = ' '.join(f'({xp:.3f} {y:.4f} {z:.4f})'
                       for z in zs for y in ys)
        sets.append(f'plane{k} {{ type cloud; points ({pts}); }}')
    txt = ('sets\n{\n' + '\n'.join(sets) + '\n}\n'
           'fields ( Q );\n'
           'interpolationScheme cellPoint;\n'
           'setFormat csv;\n'
           'timeStart %s;\ntimeEnd %s;\n' % (time, time))
    with open(os.path.join(case_dir, 'system', 'sampleDict'), 'w') as f:
        f.write('FoamFile { version 2.0; format ascii; class dictionary; '
                'object sampleDict; }\ntype sets;\n' + txt)


def latest_time(case_dir):
    ts = [d for d in os.listdir(case_dir) if d.replace('.', '', 1).isdigit()
          and d != '0' and os.path.isdir(os.path.join(case_dir, d))]
    return sorted(ts, key=float)[-1]


def case_metrics(tag, case_dir):
    t = latest_time(case_dir)
    write_surfaces_dict(case_dir)
    p = subprocess.run("openfoam -c 'postProcess -func surfaces -latestTime'",
                       shell=True, cwd=case_dir, executable='/bin/zsh',
                       capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f'surfaces failed: {(p.stderr or "")[-300:]}')
    out = {'case': tag}
    plane_dir = os.path.join(ROOT, 'results', 'lev_planes')
    os.makedirs(plane_dir, exist_ok=True)
    for k, xp in enumerate(PLANES):
        cands = glob.glob(os.path.join(case_dir, 'postProcessing', 'surfaces',
                                       f'{t}', f'p{k}.vtp'))
        if not cands:
            raise RuntimeError(f'no p{k}.vtp plane file')
        vtp = sorted(cands)[-1]
        pts = vtp_cells(vtp)
        q = vtp_field(vtp, 'Q')
        if len(q) != len(pts):  # point-data fallback
            pts = vtp_field(vtp, 'Points')
        keep = np.isfinite(q)
        pts, q = pts[keep], q[keep]
        np.savetxt(os.path.join(plane_dir, f'{tag}_p{k}.csv'), np.column_stack([pts, q]),
                   delimiter=',', header='x,y,z,Q', comments='')
        out[f'maxQ_p{k}'] = float(q.max())
        out[f'p99Q_p{k}'] = float(np.quantile(q, 0.99))
        for th in THRESHOLDS:
            out[f'fracQ{str(th).replace(".", "p")}_p{k}'] = float((q > th).mean())
    # tidy bulky surface output (CSVs preserved above)
    shutil.rmtree(os.path.join(case_dir, 'postProcessing', 'surfaces'),
                  ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', default='openfoam_template/doe_matrix_hisweep.csv')
    ap.add_argument('--runs', default='runs_hisweep')
    ap.add_argument('--cases', default=None, help='"1-20" subset')
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(ROOT, a.matrix))
    d = d[d['status'] == 'OK'].reset_index(drop=True) if 'status' in d else d
    if a.cases:
        want = set()
        for part in a.cases.split(','):
            if '-' in part:
                x, y = part.split('-', 1)
                want.update(range(int(x), int(y) + 1))
            else:
                want.add(int(part))
        d = d[[int(str(c).split('_')[-1]) in want for c in d.get(
            'case', [f'hi_{i + 1:02d}' for i in range(len(d))])]]
    rows = []
    for _, r in d.iterrows():
        tag = str(r['case']) if 'case' in r and str(r['case']).strip() else str(r.get('run_id', ''))
        case_dir = os.path.join(ROOT, a.runs, tag)
        try:
            out = case_metrics(tag, case_dir)
            out.update(sweep_deg=float(r['sweep_deg']),
                       aspect_ratio=float(r['aspect_ratio']),
                       taper_ratio=float(r['taper_ratio']))
        except Exception as e:  # noqa: BLE001 - batch must survive
            out = dict(case=tag, status=f'FAILED: {str(e)[-150:]}')
        rows.append(out)
        print(f"[{out.get('case')}] maxQ={out.get('maxQ_p0', float('nan')):.3f} "
              f"p99={out.get('p99Q_p0', float('nan')):.4f} "
              f"{out.get('status', 'OK')}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'lev_metrics.csv'),
                              index=False)
    print(f'wrote results/lev_metrics.csv ({len(rows)} rows)')


if __name__ == '__main__':
    main()
