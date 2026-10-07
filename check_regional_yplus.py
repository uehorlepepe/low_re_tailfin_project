"""Regional y+ validation over LSB zone (30-70% chord), excluding stagnation strip.
Usage: python3 check_regional_yplus.py [--case runs/case_01] [--x0 0.3 --x1 0.7]
Uses foamToVTK ASCII patch export, parses points + yPlus, maps x/c per span station.
Case params (sweep/AR/taper) read from run_batch linspace seed 42 by case index,
fallback to CLI --sweep/--ar/--taper. Writes results/<case>_regional_yplus.csv
and prints min/max/avg + percentiles for full patch vs LSB region.
"""
import argparse, base64, math, os, re, struct, subprocess, sys, zlib
import numpy as np

def vtparray(text, dtype, ncomp=1):
    raw = base64.b64decode(text.strip())
    # VTK inline binary: UInt32 header then data (uncompressed here)
    (nbytes,) = struct.unpack('<I', raw[:4])
    data = np.frombuffer(raw[4:4+nbytes], dtype=dtype)
    if ncomp > 1:
        data = data.reshape(-1, ncomp)
    return data

def parse_vtp_ascii(path, field='yPlus', ncomp=1):
    """Parse ASCII .vtp PolyData: face centres from Points+Polys, `field` from CellData."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    def find(names):
        for e in root.iter('DataArray'):
            if e.attrib.get('Name') in names:
                return np.fromstring((e.text or ''), sep=' ')
        raise KeyError(names)
    pts = find(('Points',)).reshape(-1, 3)
    conn = find(('connectivity',)).astype(int)
    offs = find(('offsets',)).astype(int)
    yp = find((field,))
    if ncomp > 1:
        yp = yp.reshape(-1, ncomp)
    # face centres
    starts = np.concatenate([[0], offs[:-1]])
    # general polygon: mean of vertices
    fc = np.empty((len(offs), 3))
    for i, (s, e) in enumerate(zip(starts, offs)):
        fc[i] = pts[conn[s:e]].mean(axis=0)
    if len(yp) == len(fc):
        vals = yp
    elif len(yp) == len(pts):
        # point data -> interpolate to faces
        vals = np.empty((len(fc), ncomp) if ncomp > 1 else len(fc))
        for i, (s, e) in enumerate(zip(starts, offs)):
            vals[i] = yp[conn[s:e]].mean(axis=0)
    else:
        raise ValueError(f'{field} len {len(yp)} matches neither faces {len(fc)} nor points {len(pts)}')
    return fc, vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='runs/case_01')
    ap.add_argument('--x0', type=float, default=0.3)
    ap.add_argument('--x1', type=float, default=0.7)
    ap.add_argument('--sweep', type=float, default=None)
    ap.add_argument('--ar', type=float, default=None)
    ap.add_argument('--taper', type=float, default=None)
    a = ap.parse_args()

    case = a.case.rstrip('/')
    # recover run_batch params by index if not given
    if a.sweep is None:
        m = re.search(r'case_(\d+)', case)
        idx = int(m.group(1)) - 1 if m else 0
        np.random.seed(42)
        n = 30
        sweeps = np.linspace(10.0, 35.0, n)
        ars = np.linspace(1.5, 3.5, n)
        tapers = np.linspace(0.4, 0.8, n)
        np.random.shuffle(ars); np.random.shuffle(tapers)
        sweep, ar, taper = float(sweeps[idx]), float(ars[idx]), float(tapers[idx])
    else:
        sweep, ar, taper = a.sweep, a.ar, a.taper

    root = 0.15; tip = root * taper
    span = ar * 0.5 * (root + tip)
    sweep_off = span * math.tan(math.radians(sweep))

    # latest time dir
    times = [d for d in os.listdir(case)
             if d.replace('.', '', 1).isdigit() and os.path.isdir(os.path.join(case, d))]
    times = [d for d in times if d != '0']
    latest = sorted(times, key=float)[-1]
    print(f'case={case} latest={latest} sweep={sweep:.2f} AR={ar:.3f} taper={taper:.3f} span={span:.3f}')

    # export ASCII patch VTK (no internal volume -> small)
    cmd = ('foamToVTK -latestTime -no-internal -patches "(tailfin)" '
           '-fields "(yPlus)" -ascii -overwrite')
    subprocess.run(f'openfoam -c \'{cmd}\'', shell=True, cwd=case,
                   executable='/bin/zsh', capture_output=True)
    # find the patch file
    cand = []
    for dp, _, fn in os.walk(os.path.join(case, 'VTK')):
        for f in fn:
            if f.endswith('.vtp') and 'tailfin' in f:
                cand.append(os.path.join(dp, f))
    if not cand:
        sys.exit('No tailfin.vtp found under VTK/ (export failed?)')
    vtp = sorted(cand)[-1]
    print('parsing', os.path.relpath(vtp))
    pts, yp = parse_vtp_ascii(vtp)
    print(f'points={len(pts)} yPlus_vals={len(yp)}')
    if len(pts) == 0 or len(yp) == 0:
        sys.exit('Parse failed (empty). Check VTK ascii output.')
    n = min(len(pts), len(yp))
    pts, yp = pts[:n], yp[:n]

    # local chord fraction per point: LE x varies with span, chord tapers
    z = pts[:, 2].clip(0, span)
    xle = sweep_off * z / span
    chord = root + (tip - root) * z / span
    xc = (pts[:, 0] - xle) / chord

    full = yp[np.isfinite(yp)]
    mask = (xc >= a.x0) & (xc <= a.x1) & np.isfinite(yp)
    reg = yp[mask]
    def stats(v):
        if len(v) == 0:
            return None
        q = np.percentile(v, [50, 95, 99])
        return dict(n=len(v), min=float(v.min()), max=float(v.max()),
                    mean=float(v.mean()), p50=float(q[0]), p95=float(q[1]), p99=float(q[2]),
                    frac_le1=float((v <= 1).mean()))
    fs, rs = stats(full), stats(reg)
    print(f'--- FULL patch --- n={fs["n"]} min={fs["min"]:.3f} max={fs["max"]:.2f} '
          f'mean={fs["mean"]:.2f} p50={fs["p50"]:.2f} p95={fs["p95"]:.2f} p99={fs["p99"]:.2f} '
          f'frac<=1={fs["frac_le1"]*100:.1f}%')
    if rs:
        print(f'--- LSB region x/c {a.x0}-{a.x1} --- n={rs["n"]} min={rs["min"]:.3f} max={rs["max"]:.2f} '
              f'mean={rs["mean"]:.2f} p50={rs["p50"]:.2f} p95={rs["p95"]:.2f} p99={rs["p99"]:.2f} '
              f'frac<=1={rs["frac_le1"]*100:.1f}%')
        print('PASS' if rs['max'] <= 1.0 else
              ('MARGINAL (p99<=1)' if rs['p99'] <= 1.0 else 'FAIL: refine layers / exclude LE strip'))
    # save per-point CSV for ParaView / log
    os.makedirs('results', exist_ok=True)
    out = os.path.join('results', os.path.basename(case) + '_regional_yplus.csv')
    np.savetxt(out, np.column_stack([pts[:n, 0], pts[:n, 1], pts[:n, 2], xc[:n], yp[:n]]),
               delimiter=',', header='x,y,z,x_over_c,yPlus', comments='')
    print('wrote', out)

if __name__ == '__main__':
    main()
