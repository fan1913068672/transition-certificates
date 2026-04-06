import json
import random
from pathlib import Path
import z3

PI = 3.1415926


def in_x(x1, x2):
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


def in_xu(x1, x2):
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def label_a(x1, x2):
    return in_xu(x1, x2)


def delta(q, x1, x2):
    if q == 1:
        return [0] if label_a(x1, x2) else [1]
    return [0]


def f(x1, x2):
    ts = 0.1
    omega = 0.01
    k = 0.0006
    import math
    return (
        x1 + ts * omega + 1.69 + ts * k * math.sin(x2 - x1) - 0.532 * x1 ** 2,
        x2 + ts * omega + 1.69 + ts * k * math.sin(x1 - x2) - 0.532 * x2 ** 2,
    )


class Template:
    # main.pdf appendix template: 7 coefficients for each (i,j), i,j in {0,1}
    def __init__(self):
        self.c = {(i, j): [z3.Real(f"c_{i}_{j}_{k}") for k in range(7)] for i in (0, 1) for j in (0, 1)}

    def T(self, i, j, x1, x2, y1, y2):
        c = self.c[(i, j)]
        I0 = z3.If(z3.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9), 1.0, 0.0)
        Iu = z3.If(z3.Or(z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9), z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9)), 1.0, 0.0)
        return (
            c[0]
            + c[1] * y1 * I0
            + c[2] * y2 * I0
            + c[3] * y1 * Iu
            + c[4] * y2 * Iu
            + c[5] * y1
            + c[6] * y2
        )


def grid(a, b, step):
    vals = []
    v = a
    while v <= b + 1e-9:
        vals.append(round(v, 8))
        v += step
    return vals


def sample_points():
    xs = grid(0.0, 8 * PI / 9, 0.5)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    xu = [(x1, x2) for x1, x2 in pts if in_xu(x1, x2)]
    return pts, x0, xu


def solve(out_file='res_cc_ex2.json'):
    pts, x0_pts, xu_pts = sample_points()
    tpl = Template()
    s = z3.SolverFor('QF_NRA')

    # keep coefficients bounded
    for coeffs in tpl.c.values():
        for ci in coeffs:
            s.add(ci >= -20, ci <= 20)

    # C1 base non-negativity on transitions
    for x1, x2 in random.sample(pts, min(120, len(pts))):
        y1, y2 = f(x1, x2)
        if not in_x(y1, y2):
            continue
        for i in (0, 1):
            for ip in delta(i, x1, x2):
                s.add(tpl.T(i, ip, x1, x2, y1, y2) >= 0)

    # C2 transitivity strengthening: T(x,z) >= T(y,z)
    z_pool = random.sample(pts, min(80, len(pts)))
    for x1, x2 in random.sample(pts, min(90, len(pts))):
        y1, y2 = f(x1, x2)
        if not in_x(y1, y2):
            continue
        for i in (0, 1):
            for ip in delta(i, x1, x2):
                for z1, z2 in z_pool[:20]:
                    for j in (0, 1):
                        s.add(
                            tpl.T(i, j, x1, x2, z1, z2)
                            >= tpl.T(ip, j, y1, y2, z1, z2)
                        )

    # C3 safety separation: init -> accepting should be <= -delta
    d = 0.1
    for x1, x2 in random.sample(x0_pts, min(20, len(x0_pts))):
        for y1, y2 in random.sample(xu_pts, min(40, len(xu_pts))):
            s.add(tpl.T(1, 0, x1, x2, y1, y2) <= -d)

    if s.check() != z3.sat:
        print('unsat')
        return

    m = s.model()
    out = {}
    for (i, j), coeffs in tpl.c.items():
        out[f'T_{i}_{j}'] = [float(m[c].as_decimal(16).replace('?', '')) if m[c] is not None else 0.0 for c in coeffs]
    Path(out_file).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved', out_file)


if __name__ == '__main__':
    solve()
