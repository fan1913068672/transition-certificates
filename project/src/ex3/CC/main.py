import argparse
import json
import random
from pathlib import Path
import z3

def f(x1, x2):
    alpha, theta, mu, Th, Te = 0.004, 0.01, 0.15, 40.0, 0.0
    def u(x):
        return 0.59 - 0.011 * x
    return (
        (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * Th * u(x1) + theta * Te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te,
    )

def in_x(x1, x2):
    return 20 <= x1 <= 34 and 20 <= x2 <= 34

def in_x0(x1, x2):
    return 21 <= x1 <= 24 and 21 <= x2 <= 24

def in_vf(x1, x2):
    return 20 <= x1 <= 26 and 20 <= x2 <= 26

def label_bits(x1, x2):
    return int(in_x0(x1, x2)), int(in_vf(x1, x2))

def delta_closure_paper(q, x1, x2):
    a0, a1 = label_bits(x1, x2)
    sigma = (a0, a1)
    if q == 0:
        if sigma == (1, 1):
            return [2]
        if sigma == (0, 1):
            return [1, 3]
        if sigma == (1, 0):
            return [1]
        return [3]
    if q == 1:
        return [2] if a1 == 1 else [1]
    if q == 2:
        return [2]
    return [3]

def delta_main_pt(q, x1, x2):
    a0, _ = label_bits(x1, x2)
    if q == 0:
        return [1] if a0 == 1 else [2]
    if q == 1:
        return [1]
    return [2]

class Template:
    # main.pdf appendix comparative setting: 11 coeffs each (i,j)
    def __init__(self, q_states):
        self.q_states = q_states
        self.c = {(i, j): [z3.Real(f"c_{i}_{j}_{k}") for k in range(11)] for i in q_states for j in q_states}

    def T(self, i, j, x1, x2, y1, y2):
        c = self.c[(i, j)]
        return (
            c[0] + c[1] * x1 + c[2] * x2 + c[3] * y1 + c[4] * y2 + c[5] * z3.If(x1 > x2, x1, x2)
            + c[6] * z3.If(y1 > y2, y1, y2) + c[7] * x1 * x1 + c[8] * x2 * x2 + c[9] * y1 * y1 + c[10] * y2 * y2
        )

def grid(a, b, step):
    vals = []
    v = a
    while v <= b + 1e-9:
        vals.append(round(v, 8))
        v += step
    return vals

def sample_points():
    xs = grid(20, 34, 1.0)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    vf = [(x1, x2) for x1, x2 in pts if in_vf(x1, x2)]
    return pts, x0, vf

def solve(mode='main', out_file='res_cc_ex3.json'):
    pts, x0_pts, vf_pts = sample_points()

    if mode == 'closure':
        q_states = [0, 1, 2, 3]
        delta = delta_closure_paper
        q0 = 0
        qacc = 2
    else:
        q_states = [0, 1, 2]
        delta = delta_main_pt
        q0 = 0
        qacc = 1

    tpl = Template(q_states)
    s = z3.SolverFor('QF_NRA')

    for coeffs in tpl.c.values():
        for ci in coeffs:
            s.add(ci >= -20, ci <= 20)

    # C1
    for x1, x2 in random.sample(pts, min(100, len(pts))):
        y1, y2 = f(x1, x2)
        if not in_x(y1, y2):
            continue
        for i in q_states:
            for ip in delta(i, x1, x2):
                s.add(tpl.T(i, ip, x1, x2, y1, y2) >= 0)

    # C2 strengthening: T(x,z) >= T(y,z)
    z_pool = random.sample(pts, min(60, len(pts)))
    for x1, x2 in random.sample(pts, min(80, len(pts))):
        y1, y2 = f(x1, x2)
        if not in_x(y1, y2):
            continue
        for i in q_states:
            for ip in delta(i, x1, x2):
                for z1, z2 in z_pool[:15]:
                    for j in q_states:
                        s.add(tpl.T(i, j, x1, x2, z1, z2) >= tpl.T(ip, j, y1, y2, z1, z2))

    # C3 persistence strengthening on accepting states
    # T(x0,y) - delta - T(x0,y') >= tau2*T(x0,y) + tau3*T(y,y')
    d, tau2, tau3 = 0.1, 0.1, 0.1
    for x01, x02 in random.sample(x0_pts, min(15, len(x0_pts))):
        for y1, y2 in random.sample(vf_pts, min(20, len(vf_pts))):
            for yp1, yp2 in random.sample(vf_pts, min(20, len(vf_pts))):
                lhs = tpl.T(q0, qacc, x01, x02, y1, y2) - d - tpl.T(q0, qacc, x01, x02, yp1, yp2)
                rhs = tau2 * tpl.T(q0, qacc, x01, x02, y1, y2) + tau3 * tpl.T(qacc, qacc, y1, y2, yp1, yp2)
                s.add(lhs >= rhs)

    if s.check() != z3.sat:
        print('unsat')
        return

    m = s.model()
    out = {'mode': mode}
    for (i, j), coeffs in tpl.c.items():
        vals = []
        for c in coeffs:
            v = m[c]
            vals.append(float(v.as_decimal(16).replace('?', '')) if v is not None else 0.0)
        out[f'T_{i}_{j}'] = vals
    Path(out_file).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved', out_file)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['main', 'closure'], default='main')
    p.add_argument('--out', default='res_cc_ex3.json')
    args = p.parse_args()
    solve(mode=args.mode, out_file=args.out)
