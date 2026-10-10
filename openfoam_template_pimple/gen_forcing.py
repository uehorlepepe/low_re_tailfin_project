#!/usr/bin/env python3
"""Dynamic forcing tables for the pimpleFoam tailfin template.

Writes constant/forcingTable [(t, U-vector)] and switches the inlet patch
to timeVaryingUniformFixedValue. Two profiles:
  sine: beta(t) = beta0 * sin(2*pi*f*t), |U| held at U_inf
  gust: lateral step of amplitude dU at t0 (effective beta step ~= dU/U_inf)

Usage (run in a case dir staged from openfoam_template_pimple):
  python3 gen_forcing.py --mode sine --beta0 10 --freq 2 --t-end 0.05
  python3 gen_forcing.py --mode gust --du 2.0 --t0 0.01 --t-end 0.05
NOTE: keyword `file` below matches the template's OpenFOAM-v2606
timeVaryingUniformFixedValue; re-verify if the container version changes.
"""
import argparse
import math
import os

U_INF = 9.0


def write_table(rows, path='constant/forcingTable'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('FoamFile { version 2.0; format ascii; '
                'class dictionary; object forcingTable; }\n(\n')
        for t, u in rows:
            f.write(f'({t:.6f} ({u[0]:.5f} {u[1]:.5f} {u[2]:.5f}))\n')
        f.write(')\n')
    print(f'wrote {path} ({len(rows)} samples)')


def patch_inlet():
    p = '0/U'
    s = open(p).read()
    new = ('    inlet\n    {\n        type            timeVaryingUniformFixedValue;\n'
           f'        file            "constant/forcingTable";\n'
           f'        refValue        uniform ({U_INF:g} 0 0);\n'
           '        outOfBounds       clamp;\n    }')
    s = re_sub_inlet(s, new)
    open(p, 'w').write(s)
    print('0/U inlet -> timeVaryingUniformFixedValue')


def re_sub_inlet(s, new):
    import re
    return re.sub(r'    inlet\s*\{[^}]*\}', new, s, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('sine', 'gust'))
    ap.add_argument('--beta0', type=float, default=10.0)
    ap.add_argument('--freq', type=float, default=2.0)
    ap.add_argument('--du', type=float, default=2.0)
    ap.add_argument('--t0', type=float, default=0.01)
    ap.add_argument('--t-end', type=float, default=0.05)
    ap.add_argument('--dt-out', type=float, default=5e-4)
    a = ap.parse_args()
    rows = []
    t = 0.0
    while t <= a.t_end + 1e-12:
        if a.mode == 'sine':
            b = math.radians(a.beta0) * math.sin(2 * math.pi * a.freq * t)
            u = (U_INF * math.cos(b), U_INF * math.sin(b), 0.0)
        else:
            uy = a.du if t >= a.t0 else 0.0
            u = (U_INF, uy, 0.0)
        rows.append((t, u))
        t += a.dt_out
    write_table(rows)
    patch_inlet()


if __name__ == '__main__':
    main()
