#!/usr/bin/env python3
"""30-case DoE batch orchestrator for the low-Re tailfin CFD pipeline.

Reads openfoam_template/doe_matrix.csv, rebuilds runs/case_XX from the frozen
template for each row (same recipe as the verified single case: coarse STL at
tol 0.1 + 0.1%-chord blunt TE + trimesh clean, level (3 4), 15 um / 12-layer
prisms), then per case: blockMesh -> surfaceFeatureExtract -> snappyHexMesh ->
simpleFoam (up to 1000 iters, early stop past a 300-iter floor once residuals
< 1e-4 and Cd/Cl flatten) -> yPlus post-process -> check_regional_yplus.py ->
forceCoeffs parse. Results accumulate in doe_results_summary.csv.

Resilience: every case runs inside try/except. A mesh/solve failure or
floating-point exception (FPE) is logged to doe_errors.log, marked FAILED in
the summary, and the sweep continues with the next case.

Usage (run from repo root so paths resolve, monitor live):
  python3 run_doe.py                         # all 30 cases, ~9 h total
  python3 run_doe.py --cases 1-3             # subset smoke test first!
  python3 run_doe.py --cases 1,5,9 --no-solve  # mesh-only check
Background:
  nohup python3 run_doe.py > doe_batch.log 2>&1 &
  tail -f doe_batch.log
  (then poll results with: column -s, -t doe_results_summary.csv | head)
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(ROOT, 'openfoam_template')
RUNS = os.path.join(ROOT, 'runs')
CAD = os.path.join(ROOT, 'cad_generator', 'build_tailfin.py')
REGIONAL = os.path.join(ROOT, 'check_regional_yplus.py')
DOE = os.path.join(TEMPLATE, 'doe_matrix.csv')
SUMMARY = os.path.join(ROOT, 'doe_results_summary.csv')
ERRLOG = os.path.join(ROOT, 'doe_errors.log')

# Frozen, verified single-case recipe (do not tune inside a sweep)
FROZEN = dict(tol=0.1, atol=0.1, te_frac=0.001, end_time=1000)
# Early-stop policy: min iters floor, residual + force stability gate
STOP = dict(min_iters=300, res_tol=1e-4, force_window=50, force_tol=1e-5,
            poll_sec=20, max_time=5400)
FPE_PATTERNS = ('FOAM FATAL', 'sigFpe', 'FOAM exiting')
# NOTE: 'Floating point' intentionally excluded - every OpenFOAM banner prints
# 'trapFpe: Floating point exception trapping enabled', a false positive.


def log(msg):
    print(f'[run_doe] {msg}', flush=True)


def errlog(msg):
    with open(ERRLOG, 'a') as f:
        f.write(msg + '\n')
    log(f'ERROR logged: {msg.splitlines()[0] if msg else ""}')


def sh(cmd, cwd, logpath=None, timeout=None):
    """Run shell cmd, stream output to logpath, return (rc, tail)."""
    p = subprocess.run(cmd, shell=True, cwd=cwd, executable='/bin/zsh',
                       capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    if logpath:
        with open(logpath, 'a') as f:
            f.write(f'\n$ {cmd}\n{out}')
    tail = '\n'.join(out.splitlines()[-5:]) if out else ''
    return p.returncode, tail


def of(cmd, cwd, logpath, timeout=None):
    """Run a containerized OpenFOAM command via the openfoam wrapper."""
    return sh(f"openfoam -c '{cmd}'", cwd, logpath, timeout)


def has_fpe(logpath):
    try:
        with open(logpath, errors='ignore') as f:
            txt = f.read()
        return any(p in txt for p in FPE_PATTERNS)
    except OSError:
        return False


RES_RE = re.compile(
    r'Solving for (Ux|Uy|Uz|p), Initial residual\s*=\s*([0-9.eE+-]+), '
    r'Final residual\s*=\s*([0-9.eE+-]+)')
TIME_RE = re.compile(r'^Time\s*=\s*(\d+)')


def read_forces_window(case_dir, window):
    """Last `window` (Cd, Cl) samples from the newest coefficient*.dat."""
    cands = glob.glob(os.path.join(case_dir, 'postProcessing', 'forceCoeffs',
                                   '*', 'coefficient*.dat'))
    if not cands:
        return []
    f = max(cands, key=os.path.getmtime)
    rows = []
    with open(f, errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                c = line.split()
                if len(c) >= 5:
                    try:
                        rows.append((float(c[1]), float(c[4])))
                    except ValueError:
                        pass
    return rows[-window:]


def check_convergence(logpath, case_dir, min_iters=300, res_tol=1e-4,
                      force_window=50, force_tol=1e-5):
    """Pure gate: (ready, iters, detail). True only past the safety floor
    with Ux/Uy/Uz/p final residuals < tol AND Cd/Cl flat over the window."""
    try:
        with open(logpath, errors='ignore') as f:
            lines = f.read().splitlines()
    except OSError:
        return False, 0, 'no log yet'
    iters = 0
    finals = {}
    for line in lines:
        m = TIME_RE.match(line.strip())
        if m:
            iters = max(iters, int(m.group(1)))
            continue
        m = RES_RE.search(line)
        if m:
            try:
                finals[m.group(1)] = float(m.group(3))  # last wins per iter
            except ValueError:
                pass
    if iters < min_iters:
        return False, iters, f'below floor ({iters}/{min_iters})'
    need = ('Ux', 'Uy', 'Uz', 'p')
    if any(v not in finals for v in need):
        return False, iters, 'residuals incomplete'
    worst = max(finals[v] for v in need)
    if worst >= res_tol:
        return False, iters, f'max final residual {worst:.2e}'
    win = read_forces_window(case_dir, force_window)
    if len(win) < force_window:
        return False, iters, f'forces warming up ({len(win)}/{force_window})'
    cds = [r[0] for r in win]
    cls = [r[1] for r in win]
    dc, dl = max(cds) - min(cds), max(cls) - min(cls)
    if dc >= force_tol or dl >= force_tol:
        return False, iters, f'forces drifting dCd={dc:.1e} dCl={dl:.1e}'
    return True, iters, (f'converged: residuals<{res_tol:.0e}, '
                         f'dCd={dc:.1e} dCl={dl:.1e}/{force_window}it')


def run_monitored_solve(case_dir, sflog, min_iters=300, res_tol=1e-4,
                        force_window=50, force_tol=1e-5,
                        poll_sec=20, max_time=5400):
    """Run simpleFoam (up to template endTime) with early termination.

    Returns (iters, stop_reason) where reason is 'converged' | 'max_iters' |
    'failed'. Kills the solver as soon as the gate passes past the floor.
    """
    import time as _time
    if os.path.exists(sflog):
        os.remove(sflog)
    with open(sflog, 'w') as lf:
        proc = subprocess.Popen(
            "openfoam -c 'simpleFoam'", shell=True, cwd=case_dir,
            executable='/bin/zsh', stdout=lf, stderr=subprocess.STDOUT)
    t0, iters, stable_polls = _time.time(), 0, 0
    try:
        while True:
            rc = proc.poll()
            if rc is not None:  # exited on its own (endTime or crash)
                if rc != 0 or has_fpe(sflog):
                    raise RuntimeError(
                        'simpleFoam failed or FPE (see log.simpleFoam)')
                return iters or FROZEN['end_time'], 'max_iters'
            if _time.time() - t0 > max_time:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError('solve wall-time budget exceeded')
            ready, iters, _ = check_convergence(
                sflog, case_dir, min_iters, res_tol, force_window, force_tol)
            if ready:
                stable_polls += 1
                if stable_polls >= 2:  # gate held across consecutive polls
                    proc.terminate()
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return iters, 'converged'
            else:
                stable_polls = 0
            _time.sleep(poll_sec)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise


def parse_cases(spec):
    """'1-30' / '1,5,9' / None -> sorted list of 1-based row indices."""
    if not spec:
        return list(range(1, 31))
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(i for i in out if 1 <= i <= 30)


def stage_case(case_dir):
    """Assemble a clean case skeleton from the frozen template."""
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, 'system'))
    os.makedirs(os.path.join(case_dir, 'constant', 'triSurface'))
    os.makedirs(os.path.join(case_dir, '0'))
    for fn in os.listdir(os.path.join(TEMPLATE, 'system')):
        shutil.copy(os.path.join(TEMPLATE, 'system', fn),
                    os.path.join(case_dir, 'system', fn))
    for fn in os.listdir(os.path.join(TEMPLATE, '0')):
        src = os.path.join(TEMPLATE, '0', fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(case_dir, '0', fn))
    for fn in ('transportProperties', 'turbulenceProperties'):
        shutil.copy(os.path.join(TEMPLATE, 'constant', fn),
                    os.path.join(case_dir, 'constant', fn))


def gen_cad(sweep, ar, taper, stl_path):
    """Regenerate STL with the frozen recipe + trimesh watertight clean."""
    cmd = (f'{sys.executable} {CAD} --sweep {sweep} --ar {ar} '
           f'--taper {taper} --out {stl_path} '
           f'--tol {FROZEN["tol"]} --atol {FROZEN["atol"]} --te-frac {FROZEN["te_frac"]}')
    p = subprocess.run(cmd, shell=True, executable='/bin/zsh',
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f'CAD generation failed: {(p.stderr or "")[-500:]}')
    clean = (
        'import trimesh; m = trimesh.load("%s", force="mesh"); '
        'm.merge_vertices(); m.update_faces(m.nondegenerate_faces()); '
        'm.remove_unreferenced_vertices(); trimesh.repair.fix_winding(m); '
        'trimesh.repair.fix_normals(m); '
        'm.export("%s", file_type="stl"); '
        'assert m.is_watertight, "STL not watertight"; print("faces:", len(m.faces))'
        % (stl_path, stl_path))
    p = subprocess.run(f'{sys.executable} -c \'{clean}\'', shell=True,
                       executable='/bin/zsh', capture_output=True,
                       text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f'STL clean failed: {(p.stderr or "")[-500:]}')
    return (p.stdout or '').strip()


def parse_forces(case_dir):
    """Last row of newest coefficient*.dat -> (Cd, Cl, Cs)."""
    cands = glob.glob(os.path.join(case_dir, 'postProcessing', 'forceCoeffs',
                                   '*', 'coefficient*.dat'))
    if not cands:
        raise RuntimeError('no coefficient.dat produced')
    f = max(cands, key=os.path.getmtime)
    last = None
    with open(f, errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                last = line.split()
    if not last or len(last) < 11:
        raise RuntimeError(f'unparseable coefficient row in {f}')
    return float(last[1]), float(last[4]), float(last[10]), os.path.basename(f)


def parse_yplus_output(text):
    """'patch tailfin y+ : min = a, max = b, average = c' -> tuple."""
    m = re.search(r'min\s*=\s*([0-9.eE+-]+),\s*max\s*=\s*([0-9.eE+-]+),\s*average\s*=\s*([0-9.eE+-]+)',
                  text or '')
    if not m:
        raise RuntimeError('yPlus summary line not found in postProcess output')
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_regional_output(text):
    """Grab FULL + LSB stats lines from check_regional_yplus.py stdout."""
    stats = {}
    for line in (text or '').splitlines():
        m = re.match(r'--- (FULL patch|LSB region x/c [0-9.]+-[0-9.]+) --- n=(\d+) '
                     r'min=([0-9.eE+-]+) max=([0-9.eE+-]+) mean=([0-9.eE+-]+) '
                     r'p50=([0-9.eE+-]+) p95=([0-9.eE+-]+) p99=([0-9.eE+-]+) '
                     r'frac<=1=([0-9.]+)%', line)
        if m:
            key = 'full' if m.group(1) == 'FULL patch' else 'lsb'
            stats[key] = dict(n=int(m.group(2)), min=float(m.group(3)),
                              max=float(m.group(4)), mean=float(m.group(5)),
                              p50=float(m.group(6)), p95=float(m.group(7)),
                              p99=float(m.group(8)), frac_le1=float(m.group(9)) / 100.0)
    if 'full' not in stats or 'lsb' not in stats:
        raise RuntimeError('regional stats lines missing from extractor output')
    return stats


def run_case(idx, row, no_solve=False):
    """Full pipeline for one DoE row. Returns result dict; raises on failure."""
    run_id = (row.get('run_id') or f'run_{idx:02d}').strip()
    sweep = float(row['sweep_deg'])
    taper = float(row['taper_ratio'])
    ar = float(row['aspect_ratio'])
    case_dir = os.path.join(RUNS, f'case_{idx:02d}')
    res = dict(run_id=run_id, case=f'case_{idx:02d}', sweep_deg=sweep,
               aspect_ratio=ar, taper_ratio=taper, status='OK')
    log(f'===== {res["case"]} ({run_id}): sweep={sweep} AR={ar} taper={taper} =====')

    # 1. stage + CAD
    stage_case(case_dir)
    stl = os.path.join(case_dir, 'constant', 'triSurface', 'tailfin.stl')
    res['stl_info'] = gen_cad(sweep, ar, taper, stl)
    log(f'{res["case"]}: STL ok ({res["stl_info"]})')

    slog = os.path.join(case_dir, 'log.snappyHexMesh')
    if os.path.exists(slog):
        os.remove(slog)
    # 2. mesh
    rc, tail = of('blockMesh > snappy_sc.log 2>&1 && rm -f '
                  'constant/triSurface/tailfin.eMesh && surfaceFeatureExtract '
                  '>> snappy_sc.log 2>&1 && snappyHexMesh -overwrite '
                  '>> snappy_sc.log 2>&1', case_dir, None, timeout=1500)
    with open(os.path.join(case_dir, 'snappy_sc.log'), errors='ignore') as f:
        mtxt = f.read()
    cov = re.findall(r'Extruding\s+(\d+)\s+out of\s+(\d+)', mtxt)
    res['layer_coverage'] = (f'{int(cov[-1][0]) / int(cov[-1][1]) * 100:.1f}%'
                             if cov else 'unknown')
    if rc != 0 or 'Finished meshing' not in mtxt or has_fpe(
            os.path.join(case_dir, 'snappy_sc.log')):
        raise RuntimeError(f'meshing failed [{res["layer_coverage"]} coverage]: {tail[-300:]}')
    cells = re.findall(r'Layer mesh\s*:\s*cells\s*:\s*(\d+)', mtxt)
    res['cells'] = int(cells[-1]) if cells else -1
    log(f'{res["case"]}: meshed ({res["cells"]} cells, {res["layer_coverage"]} layers)')

    if no_solve:
        res['status'] = 'MESH_ONLY'
        return res

    # 3. solve (1000-iter harbor, early stop past the 300-iter floor)
    sflog = os.path.join(case_dir, 'log.simpleFoam')
    iters, reason = run_monitored_solve(
        case_dir, sflog, min_iters=STOP['min_iters'], res_tol=STOP['res_tol'],
        force_window=STOP['force_window'], force_tol=STOP['force_tol'],
        poll_sec=STOP['poll_sec'], max_time=STOP['max_time'])
    res['iters'] = iters
    res['stop_reason'] = reason
    log(f'{res["case"]}: solved ({iters} iters, {reason})')

    # 4. yPlus summary
    rc, tail = of('simpleFoam -postProcess -func yPlus -latestTime',
                  case_dir, None, timeout=600)
    ymn, ymx, yav = parse_yplus_output(tail)
    res.update(yplus_min=ymn, yplus_max=ymx, yplus_avg=yav)

    # 5. regional extraction (passes true DoE params explicitly)
    p = subprocess.run(
        f'{sys.executable} {REGIONAL} --case runs/case_{idx:02d} '
        f'--sweep {sweep} --ar {ar} --taper {taper}',
        shell=True, cwd=ROOT, executable='/bin/zsh', capture_output=True,
        text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError(f'regional extractor failed: {(p.stderr or "")[-300:]}')
    stats = parse_regional_output(p.stdout or '')
    res.update(lsb_mean=stats['lsb']['mean'], lsb_max=stats['lsb']['max'],
               lsb_p95=stats['lsb']['p95'], lsb_frac_le1=stats['lsb']['frac_le1'])
    log(f'{res["case"]}: y+ avg={yav:.2f} max={ymx:.1f} | '
        f'LSB mean={res["lsb_mean"]:.2f} max={res["lsb_max"]:.1f}')

    # 6. forces (Cd, Cl, Cs->Cy)
    cd, cl, cs, _ = parse_forces(case_dir)
    res.update(Cd=cd, Cl=cl, Cy=cs)
    log(f'{res["case"]}: Cd={cd:.5f} Cl={cl:.5f} Cy={cs:.6f}')

    # 7. tidy bulky artifacts (keep final time dir for audit)
    for bulky in glob.glob(os.path.join(case_dir, 'VTK')):
        shutil.rmtree(bulky, ignore_errors=True)
    times = sorted((d for d in os.listdir(case_dir)
                    if d.replace('.', '', 1).isdigit() and d != '0'
                    and os.path.isdir(os.path.join(case_dir, d))),
                   key=float)
    for t in times[:-1]:
        shutil.rmtree(os.path.join(case_dir, t), ignore_errors=True)
    return res


COLUMNS = ['run_id', 'case', 'sweep_deg', 'aspect_ratio', 'taper_ratio',
           'status', 'cells', 'layer_coverage', 'iters', 'stop_reason',
           'Cd', 'Cl', 'Cy',
           'yplus_min', 'yplus_max', 'yplus_avg',
           'lsb_mean', 'lsb_max', 'lsb_p95', 'lsb_frac_le1', 'notes']


def append_row(res):
    new = not os.path.exists(SUMMARY)
    with open(SUMMARY, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: res.get(k, '') for k in COLUMNS})


def main():
    ap = argparse.ArgumentParser(description='30-case tailfin DoE batch sweep')
    ap.add_argument('--cases', default=None, help='"1-30", "1,5,9", default all')
    ap.add_argument('--no-solve', action='store_true', help='mesh-only smoke test')
    a = ap.parse_args()

    # Preflight: CAD + STL steps need the HOST python env (cadquery, trimesh).
    # Inside the `openfoam` container shell these are missing -> exit early.
    try:
        import trimesh  # noqa: F401
        import cadquery  # noqa: F401
    except ImportError as e:
        sys.exit(f'PREFLIGHT FAILED: {e}. Run from a normal terminal '
                 f'(host python3), NOT inside the `openfoam` shell. '
                 f'Type `exit`, then: python3 run_doe.py ...')

    with open(DOE, newline='') as f:
        rows = list(csv.DictReader(f))
    idxs = [i for i in parse_cases(a.cases) if i <= len(rows)]
    log(f'{len(idxs)} case(s) queued (python {sys.version.split()[0]}). '
        f'Summary -> doe_results_summary.csv, errors -> doe_errors.log')

    for n, idx in enumerate(idxs, 1):
        try:
            res = run_case(idx, rows[idx - 1], no_solve=a.no_solve)
            res['notes'] = '' if res['status'] == 'OK' else res['status']
        except subprocess.TimeoutExpired:
            res = dict(run_id=rows[idx - 1].get('run_id', f'run_{idx:02d}'),
                       case=f'case_{idx:02d}',
                       sweep_deg=rows[idx - 1].get('sweep_deg', ''),
                       aspect_ratio=rows[idx - 1].get('aspect_ratio', ''),
                       taper_ratio=rows[idx - 1].get('taper_ratio', ''),
                       status='FAILED', notes='timeout')
            em = f"[{res['case']}] TIMEOUT after step limit"
            errlog(em + '\n' + traceback.format_exc()[-800:])
        except Exception as e:  # noqa: BLE001 - batch must survive any case failure
            res = dict(run_id=rows[idx - 1].get('run_id', f'run_{idx:02d}'),
                       case=f'case_{idx:02d}',
                       sweep_deg=rows[idx - 1].get('sweep_deg', ''),
                       aspect_ratio=rows[idx - 1].get('aspect_ratio', ''),
                       taper_ratio=rows[idx - 1].get('taper_ratio', ''),
                       status='FAILED', notes=str(e)[-300:])
            em = f"[{res['case']}] {type(e).__name__}: {e}"
            errlog(em + '\n' + traceback.format_exc()[-800:])
        append_row(res)
        log(f'progress {n}/{len(idxs)} -> {res["status"]} ({res["case"]})')
    log(f'done. Summary: {SUMMARY}')


if __name__ == '__main__':
    main()
