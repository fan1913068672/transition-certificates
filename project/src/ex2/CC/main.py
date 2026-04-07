import json
import math
import random
from pathlib import Path
import argparse
import time
import sys

import dreal
import z3

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

PI = 3.1415926


def in_x(x1, x2):
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


def in_xu(x1, x2):
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def in_x_cond(x1, x2):
    return dreal.And(x1 >= 0, x1 <= 8 * PI / 9, x2 >= 0, x2 <= 8 * PI / 9)


def in_x0_cond(x1, x2):
    return dreal.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9)


def in_xu_cond(x1, x2):
    return dreal.Or(
        dreal.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
        dreal.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
    )


def f_num(x1, x2):
    ts, omega, k = 0.1, 0.01, 0.0006
    return (
        x1 + ts * omega + 1.69 + ts * k * math.sin(x2 - x1) - 0.532 * x1**2,
        x2 + ts * omega + 1.69 + ts * k * math.sin(x1 - x2) - 0.532 * x2**2,
    )


def f_sym(x1, x2):
    ts, omega, k = 0.1, 0.01, 0.0006
    return (
        x1 + ts * omega + 1.69 + ts * k * dreal.sin(x2 - x1) - 0.532 * x1**2,
        x2 + ts * omega + 1.69 + ts * k * dreal.sin(x1 - x2) - 0.532 * x2**2,
    )


def delta(q, x1, x2):
    if q == 1:
        return [0] if in_xu(x1, x2) else [1]
    return [0]


class Template:
    """main appendix ex2 CC template (7 coefficients per (i,j))."""

    def __init__(self):
        self.c = {(i, j): [z3.Real(f"c_{i}_{j}_{k}") for k in range(7)] for i in (0, 1) for j in (0, 1)}

    def t_z3(self, i, j, x1, x2, y1, y2):
        c = self.c[(i, j)]
        i0 = z3.If(z3.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9), 1.0, 0.0)
        iu = z3.If(
            z3.Or(z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9), z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9)),
            1.0,
            0.0,
        )
        return c[0] + c[1] * y1 * i0 + c[2] * y2 * i0 + c[3] * y1 * iu + c[4] * y2 * iu + c[5] * y1 + c[6] * y2

    def t_dreal(self, coeffs, i, j, x1, x2, y1, y2):
        c = coeffs[(i, j)]
        i0 = dreal.if_then_else(in_x0_cond(x1, x2), 1.0, 0.0)
        iu = dreal.if_then_else(in_xu_cond(x1, x2), 1.0, 0.0)
        return c[0] + c[1] * y1 * i0 + c[2] * y2 * i0 + c[3] * y1 * iu + c[4] * y2 * iu + c[5] * y1 + c[6] * y2


def grid(a, b, step):
    vals = []
    v = a
    while v <= b + 1e-9:
        vals.append(round(v, 8))
        v += step
    return vals


def t2float(v, precision=16):
    s = v.as_decimal(precision)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def mid_float(iv):
    return float(iv.mid())


def synthesize(max_iter=60, out_file='res_cc_ex2.json'):
    start = time.time()
    tpl = Template()
    s = z3.SolverFor('QF_NRA')

    for coeffs in tpl.c.values():
        for ci in coeffs:
            s.add(ci >= -20, ci <= 20)

    eps = z3.Real('eps')
    s.add(eps > 0, eps <= 5)

    xs = grid(0.0, 8 * PI / 9, 0.5)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]

    # initial sampled constraints for faster start
    for x1, x2 in random.sample(pts, min(120, len(pts))):
        x1n, x2n = f_num(x1, x2)
        if not in_x(x1n, x2n):
            continue
        for i in (0, 1):
            for ip in delta(i, x1, x2):
                s.add(tpl.t_z3(i, ip, x1, x2, x1n, x2n) >= 0)

    for x1, x2 in random.sample(x0, min(20, len(x0))):
        for z1, z2 in random.sample(pts, min(30, len(pts))):
            p1 = tpl.t_z3(1, 0, x1, x2, z1, z2) >= 0
            p2 = tpl.t_z3(0, 0, z1, z2, z1, z2) >= 0
            c3 = tpl.t_z3(1, 0, x1, x2, z1, z2) >= tpl.t_z3(1, 0, x1, x2, z1, z2) + eps
            s.add(z3.Implies(z3.And(p1, p2), z3.Not(c3)))

    # CEGIS with dReal counterexamples
    for it in range(max_iter):
        if s.check() != z3.sat:
            print('unsat')
            return None

        m = s.model()
        coeff_vals = {(i, j): [t2float(m[c]) for c in tpl.c[(i, j)]] for i in (0, 1) for j in (0, 1)}
        eps_val = t2float(m[eps])

        ce = find_counterexample(coeff_vals, eps_val, precision=1e-4)
        if ce is None:
            out = {'eps': eps_val}
            for (i, j), vv in coeff_vals.items():
                out[f'T_{i}_{j}'] = vv
            out["success"] = True
            out["iterations"] = it + 1
            out["elapsed_sec"] = time.time() - start
            Path(out_file).write_text(json.dumps(out, indent=2), encoding='utf-8')
            print('saved', out_file)
            return out

        kind = ce['kind']
        if kind == 'c1':
            x1, x2, i, ip = ce['x1'], ce['x2'], ce['i'], ce['ip']
            x1n, x2n = f_num(x1, x2)
            s.add(tpl.t_z3(i, ip, x1, x2, x1n, x2n) >= 0)
        elif kind == 'c2':
            x1, x2, y1, y2, i, j, ip = ce['x1'], ce['x2'], ce['y1'], ce['y2'], ce['i'], ce['j'], ce['ip']
            premise = tpl.t_z3(ip, j, f_num(x1, x2)[0], f_num(x1, x2)[1], y1, y2) >= 0
            concl = tpl.t_z3(i, j, x1, x2, y1, y2) >= 0
            s.add(z3.Implies(premise, concl))
        else:
            x01, x02, z1, z2, zp1, zp2 = ce['x01'], ce['x02'], ce['z1'], ce['z2'], ce['zp1'], ce['zp2']
            p1 = tpl.t_z3(1, 0, x01, x02, z1, z2) >= 0
            p2 = tpl.t_z3(0, 0, z1, z2, zp1, zp2) >= 0
            c = tpl.t_z3(1, 0, x01, x02, zp1, zp2) + eps <= tpl.t_z3(1, 0, x01, x02, z1, z2)
            s.add(z3.Implies(z3.And(p1, p2), c))

        print(f'iter={it+1}, added {kind} counterexample')

    print('max_iter reached')
    return {"success": False, "iterations": max_iter, "elapsed_sec": time.time() - start}


def find_counterexample(coeffs, eps_val, precision=1e-4):
    tpl = Template()

    # C1: T((x,i),(f(x),ip)) >= 0
    for i in (0, 1):
        x1 = dreal.Variable('x1')
        x2 = dreal.Variable('x2')
        ctx = dreal.Context()
        ctx.SetLogic(dreal.Logic.QF_NRA)
        ctx.config.precision = precision
        ctx.DeclareVariable(x1, 0, 8 * PI / 9)
        ctx.DeclareVariable(x2, 0, 8 * PI / 9)
        ctx.Assert(in_x_cond(x1, x2))
        x1n, x2n = f_sym(x1, x2)
        for ip, cond in ((0, in_xu_cond(x1, x2)), (1, dreal.Not(in_xu_cond(x1, x2)))) if i == 1 else ((0, None),):
            ctx.Push(1)
            if cond is not None:
                ctx.Assert(cond)
            ctx.Assert(tpl.t_dreal(coeffs, i, ip, x1, x2, x1n, x2n) < -1e-8)
            m = ctx.CheckSat()
            if m is not None:
                return {'kind': 'c1', 'x1': mid_float(m[x1]), 'x2': mid_float(m[x2]), 'i': i, 'ip': ip}
            ctx.Pop(1)

    # C2
    for i in (0, 1):
        for j in (0, 1):
            x1 = dreal.Variable('x1')
            x2 = dreal.Variable('x2')
            y1 = dreal.Variable('y1')
            y2 = dreal.Variable('y2')
            ctx = dreal.Context()
            ctx.SetLogic(dreal.Logic.QF_NRA)
            ctx.config.precision = precision
            ctx.DeclareVariable(x1, 0, 8 * PI / 9)
            ctx.DeclareVariable(x2, 0, 8 * PI / 9)
            ctx.DeclareVariable(y1, 0, 8 * PI / 9)
            ctx.DeclareVariable(y2, 0, 8 * PI / 9)
            ctx.Assert(in_x_cond(x1, x2))
            ctx.Assert(in_x_cond(y1, y2))
            x1n, x2n = f_sym(x1, x2)
            if i == 1:
                branches = [(0, in_xu_cond(x1, x2)), (1, dreal.Not(in_xu_cond(x1, x2)))]
            else:
                branches = [(0, None)]
            for ip, cond in branches:
                ctx.Push(1)
                if cond is not None:
                    ctx.Assert(cond)
                premise = tpl.t_dreal(coeffs, ip, j, x1n, x2n, y1, y2) >= 0
                bad = tpl.t_dreal(coeffs, i, j, x1, x2, y1, y2) < -1e-8
                ctx.Assert(dreal.And(premise, bad))
                m = ctx.CheckSat()
                if m is not None:
                    return {
                        'kind': 'c2', 'x1': mid_float(m[x1]), 'x2': mid_float(m[x2]), 'y1': mid_float(m[y1]), 'y2': mid_float(m[y2]),
                        'i': i, 'j': j, 'ip': ip
                    }
                ctx.Pop(1)

    # C3
    x01 = dreal.Variable('x01')
    x02 = dreal.Variable('x02')
    z1 = dreal.Variable('z1')
    z2 = dreal.Variable('z2')
    zp1 = dreal.Variable('zp1')
    zp2 = dreal.Variable('zp2')
    ctx = dreal.Context()
    ctx.SetLogic(dreal.Logic.QF_NRA)
    ctx.config.precision = precision
    ctx.DeclareVariable(x01, 0, PI / 9)
    ctx.DeclareVariable(x02, 0, PI / 9)
    ctx.DeclareVariable(z1, 0, 8 * PI / 9)
    ctx.DeclareVariable(z2, 0, 8 * PI / 9)
    ctx.DeclareVariable(zp1, 0, 8 * PI / 9)
    ctx.DeclareVariable(zp2, 0, 8 * PI / 9)
    p1 = tpl.t_dreal(coeffs, 1, 0, x01, x02, z1, z2) >= 0
    p2 = tpl.t_dreal(coeffs, 0, 0, z1, z2, zp1, zp2) >= 0
    bad = tpl.t_dreal(coeffs, 1, 0, x01, x02, zp1, zp2) + eps_val > tpl.t_dreal(coeffs, 1, 0, x01, x02, z1, z2)
    ctx.Assert(dreal.And(in_x0_cond(x01, x02), in_x_cond(z1, z2), in_x_cond(zp1, zp2), p1, p2, bad))
    m = ctx.CheckSat()
    if m is not None:
        return {
            'kind': 'c3',
            'x01': mid_float(m[x01]), 'x02': mid_float(m[x02]), 'z1': mid_float(m[z1]), 'z2': mid_float(m[z2]),
            'zp1': mid_float(m[zp1]), 'zp2': mid_float(m[zp2])
        }

    return None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ex2 closure-certificate synthesis")
    p.add_argument("--out", type=str, default="res_cc_ex2.json")
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--grid-step", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    print_header("ex2", "CC", "closure_certificate", {"max_iter": args.max_iter, "solver_synth": "z3", "solver_verify": "dreal"})
    result = synthesize(max_iter=args.max_iter, out_file=str(out_path))
    if result is None:
        result = {"success": False, "elapsed_sec": 0.0}
    if not out_path.exists():
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))
