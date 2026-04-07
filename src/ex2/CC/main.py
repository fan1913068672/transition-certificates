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
# S-procedure coefficients (Closure Certificates, Eq. (24)-(25)-style strengthening)
TAU1 = 1.0  # for C2 strengthening
TAU2 = 1.0  # for C3 strengthening
TAU3 = 0.0  # for C3 strengthening
LOCAL_H = 0.02  # local fine-grained sampling step around counterexamples


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


def local_neighbors_2d(x1, x2, lo, hi, h=LOCAL_H):
    pts = []
    for dx in (-h, 0.0, h):
        for dy in (-h, 0.0, h):
            xx1 = max(lo, min(hi, x1 + dx))
            xx2 = max(lo, min(hi, x2 + dy))
            pts.append((round(xx1, 8), round(xx2, 8)))
    # deduplicate after clipping
    return list(dict.fromkeys(pts))


def t2float(v, precision=16):
    s = v.as_decimal(precision)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def t_num(coeffs, i, j, x1, x2, y1, y2):
    c = coeffs[(i, j)]
    i0 = 1.0 if in_x0(x1, x2) else 0.0
    iu = 1.0 if in_xu(x1, x2) else 0.0
    return c[0] + c[1] * y1 * i0 + c[2] * y2 * i0 + c[3] * y1 * iu + c[4] * y2 * iu + c[5] * y1 + c[6] * y2


def mid_float(iv):
    return float(iv.mid())


def synthesize(max_iter=60, grid_step=0.1, out_file='res_cc_ex2.json'):
    start = time.time()
    tpl = Template()
    s = z3.SolverFor('QF_NRA')

    for coeffs in tpl.c.values():
        for ci in coeffs:
            s.add(ci >= -20, ci <= 20)

    eps = z3.Real('eps')
    s.add(eps > 0, eps <= 5)

    if grid_step <= 0:
        raise ValueError("grid_step must be > 0")
    xs = grid(0.0, 8 * PI / 9, grid_step)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    xu = [(x1, x2) for x1, x2 in pts if in_xu(x1, x2)]
    print(f"Dataset sizes: |X|={len(pts)}, |X0|={len(x0)}, |Xu|={len(xu)}, grid_step={grid_step}")

    # initial sampled constraints for faster start
    c1_seed_n = min(120, len(pts))
    for x1, x2 in random.sample(pts, c1_seed_n):
        x1n, x2n = f_num(x1, x2)
        if not in_x(x1n, x2n):
            continue
        for i in (0, 1):
            for ip in delta(i, x1, x2):
                s.add(tpl.t_z3(i, ip, x1, x2, x1n, x2n) >= 0)

    # sampled C2 strengthening:
    # TAU1 * T_{i',j}(f(x), y) <= T_{i,j}(x, y)
    y_pool_n = min(80, len(pts))
    c2_x_seed_n = min(80, len(pts))
    y_pool = random.sample(pts, y_pool_n)
    for x1, x2 in random.sample(pts, c2_x_seed_n):
        x1n, x2n = f_num(x1, x2)
        if not in_x(x1n, x2n):
            continue
        for i in (0, 1):
            for ip in delta(i, x1, x2):
                for y1, y2 in y_pool[:20]:
                    for j in (0, 1):
                        s.add(TAU1 * tpl.t_z3(ip, j, x1n, x2n, y1, y2) <= tpl.t_z3(i, j, x1, x2, y1, y2))

    c3_x0_seed_n = min(20, len(x0))
    c3_z_seed_n = min(30, len(pts))
    for x1, x2 in random.sample(x0, c3_x0_seed_n):
        for z1, z2 in random.sample(pts, c3_z_seed_n):
            # C3 sampled seed:
            # T_10(x0,z) - eps - T_10(x0,z') >= TAU2*T_10(x0,z) + TAU3*T_00(z,z')
            zp1, zp2 = f_num(z1, z2)
            if not in_x(zp1, zp2):
                continue
            lhs = tpl.t_z3(1, 0, x1, x2, z1, z2) - eps - tpl.t_z3(1, 0, x1, x2, zp1, zp2)
            rhs = TAU2 * tpl.t_z3(1, 0, x1, x2, z1, z2) + TAU3 * tpl.t_z3(0, 0, z1, z2, zp1, zp2)
            s.add(lhs >= rhs)
    print(f"Seed constraints: C1={c1_seed_n}, C2≈{c2_x_seed_n}x{min(20, y_pool_n)}x4, C3≈{c3_x0_seed_n}x{c3_z_seed_n}")

    blocked_c2 = set()
    # CEGIS with dReal counterexamples (batch style, ex1/CC-like)
    for it in range(max_iter):
        print(f"Dataset sizes: |X|={len(pts)}, |X0|={len(x0)}, |Xu|={len(xu)}, grid_step={grid_step}")
        if s.check() != z3.sat:
            print('unsat')
            return None

        m = s.model()
        coeff_vals = {(i, j): [t2float(m[c]) for c in tpl.c[(i, j)]] for i in (0, 1) for j in (0, 1)}
        eps_val = t2float(m[eps])
        print(f"iter={it+1}, candidate eps={eps_val:.6f}")

        ce_list = find_counterexamples(coeff_vals, eps_val, precision=1e-4, max_per_kind=3, blocked_c2=blocked_c2)
        if len(ce_list) == 0:
            out = {'eps': eps_val}
            for (i, j), vv in coeff_vals.items():
                out[f'T_{i}_{j}'] = vv
            out["success"] = True
            out["iterations"] = it + 1
            out["elapsed_sec"] = time.time() - start
            Path(out_file).write_text(json.dumps(out, indent=2), encoding='utf-8')
            print('saved', out_file)
            return out

        c1_add, c2_add, c3_add = 0, 0, 0
        c2_local_add = 0
        c2_examples = []
        for ce in ce_list:
            kind = ce['kind']
            if kind == 'c1':
                x1, x2, i, ip = ce['x1'], ce['x2'], ce['i'], ce['ip']
                x1n, x2n = f_num(x1, x2)
                s.add(tpl.t_z3(i, ip, x1, x2, x1n, x2n) >= 0)
                c1_add += 1
            elif kind == 'c2':
                x1, x2, y1, y2, i, j, ip = ce['x1'], ce['x2'], ce['y1'], ce['y2'], ce['i'], ce['j'], ce['ip']
                blocked_c2.add((round(x1, 6), round(x2, 6), round(y1, 6), round(y2, 6), i, j, ip))
                fx1, fx2 = f_num(x1, x2)
                # S-procedure strengthened C2
                s.add(TAU1 * tpl.t_z3(ip, j, fx1, fx2, y1, y2) <= tpl.t_z3(i, j, x1, x2, y1, y2))
                c2_add += 1
                c2_examples.append((x1, x2, y1, y2, i, j, ip))
                # local fine-grained sampling around C2 counterexample
                x_nbr = local_neighbors_2d(x1, x2, 0.0, 8 * PI / 9)
                y_nbr = local_neighbors_2d(y1, y2, 0.0, 8 * PI / 9)
                for xx1, xx2 in x_nbr:
                    fxx1, fxx2 = f_num(xx1, xx2)
                    if not in_x(fxx1, fxx2):
                        continue
                    for yy1, yy2 in y_nbr:
                        s.add(TAU1 * tpl.t_z3(ip, j, fxx1, fxx2, yy1, yy2) <= tpl.t_z3(i, j, xx1, xx2, yy1, yy2))
                        c2_local_add += 1
            else:
                x01, x02, z1, z2, zp1, zp2 = ce['x01'], ce['x02'], ce['z1'], ce['z2'], ce['zp1'], ce['zp2']
                # S-procedure strengthened C3
                lhs = tpl.t_z3(1, 0, x01, x02, z1, z2) - eps - tpl.t_z3(1, 0, x01, x02, zp1, zp2)
                rhs = TAU2 * tpl.t_z3(1, 0, x01, x02, z1, z2) + TAU3 * tpl.t_z3(0, 0, z1, z2, zp1, zp2)
                s.add(lhs >= rhs)
                c3_add += 1

        print(f'iter={it+1}, added CE batch: c1={c1_add}, c2={c2_add}, c3={c3_add}, c2_local={c2_local_add}, total={len(ce_list)}')
        if c2_examples:
            x1, x2, y1, y2, i, j, ip = c2_examples[0]
            fx1, fx2 = f_num(x1, x2)
            c2_gap = t_num(coeff_vals, i, j, x1, x2, y1, y2) - TAU1 * t_num(coeff_vals, ip, j, fx1, fx2, y1, y2)
            print(
                f"  c2-ce: (x1={x1:.6f}, x2={x2:.6f}, y1={y1:.6f}, y2={y2:.6f}, i={i}, j={j}, ip={ip})"
            )
            print(f"  c2-gap = {c2_gap:.12f}  (need >= 0)")

    print('max_iter reached')
    return {"success": False, "iterations": max_iter, "elapsed_sec": time.time() - start}


def _append_unique(collector, ce, key_set, key):
    if key not in key_set:
        key_set.add(key)
        collector.append(ce)


def find_counterexamples(coeffs, eps_val, precision=1e-4, max_per_kind=3, blocked_c2=None):
    tpl = Template()
    ce_all = []
    if blocked_c2 is None:
        blocked_c2 = set()

    # C1: T((x,i),(f(x),ip)) >= 0
    ce_c1 = []
    seen_c1 = set()
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
                x1m = mid_float(m[x1]); x2m = mid_float(m[x2])
                key = (round(x1m, 6), round(x2m, 6), i, ip)
                _append_unique(ce_c1, {'kind': 'c1', 'x1': x1m, 'x2': x2m, 'i': i, 'ip': ip}, seen_c1, key)
                if len(ce_c1) >= max_per_kind:
                    ce_all.extend(ce_c1)
                    break
            ctx.Pop(1)
        if len(ce_c1) >= max_per_kind:
            break
    ce_all.extend(ce_c1)

    # C2
    ce_c2 = []
    seen_c2 = set()
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
            # block previously seen C2 counterexamples to avoid repeating the same witness
            block_r = 1e-4
            for bx1, bx2, by1, by2, bi, bj, bip in blocked_c2:
                if bi == i and bj == j:
                    disj = dreal.Or(
                        x1 <= bx1 - block_r, x1 >= bx1 + block_r,
                        x2 <= bx2 - block_r, x2 >= bx2 + block_r,
                        y1 <= by1 - block_r, y1 >= by1 + block_r,
                        y2 <= by2 - block_r, y2 >= by2 + block_r,
                    )
                    if i == 1:
                        if bip == 0:
                            ctx.Assert(dreal.Or(dreal.Not(in_xu_cond(x1, x2)), disj))
                        else:
                            ctx.Assert(dreal.Or(in_xu_cond(x1, x2), disj))
                    else:
                        ctx.Assert(disj)
            x1n, x2n = f_sym(x1, x2)
            if i == 1:
                branches = [(0, in_xu_cond(x1, x2)), (1, dreal.Not(in_xu_cond(x1, x2)))]
            else:
                branches = [(0, None)]
            for ip, cond in branches:
                ctx.Push(1)
                if cond is not None:
                    ctx.Assert(cond)
                # violation of strengthened C2: TAU1*T(ip,j,f(x),y) <= T(i,j,x,y)
                bad = tpl.t_dreal(coeffs, i, j, x1, x2, y1, y2) - TAU1 * tpl.t_dreal(coeffs, ip, j, x1n, x2n, y1, y2) < -1e-8
                ctx.Assert(bad)
                m = ctx.CheckSat()
                if m is not None:
                    x1m = mid_float(m[x1]); x2m = mid_float(m[x2]); y1m = mid_float(m[y1]); y2m = mid_float(m[y2])
                    key = (round(x1m, 6), round(x2m, 6), round(y1m, 6), round(y2m, 6), i, j, ip)
                    _append_unique(ce_c2, {
                        'kind': 'c2', 'x1': mid_float(m[x1]), 'x2': mid_float(m[x2]), 'y1': mid_float(m[y1]), 'y2': mid_float(m[y2]),
                        'i': i, 'j': j, 'ip': ip
                    }, seen_c2, key)
                    if len(ce_c2) >= max_per_kind:
                        ce_all.extend(ce_c2)
                        break
                ctx.Pop(1)
            if len(ce_c2) >= max_per_kind:
                break
        if len(ce_c2) >= max_per_kind:
            break
    ce_all.extend(ce_c2)

    # C3
    ce_c3 = []
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
    # violation of strengthened C3:
    # T10(x0,z)-eps-T10(x0,z') >= TAU2*T10(x0,z)+TAU3*T00(z,z')
    lhs = tpl.t_dreal(coeffs, 1, 0, x01, x02, z1, z2) - eps_val - tpl.t_dreal(coeffs, 1, 0, x01, x02, zp1, zp2)
    rhs = TAU2 * tpl.t_dreal(coeffs, 1, 0, x01, x02, z1, z2) + TAU3 * tpl.t_dreal(coeffs, 0, 0, z1, z2, zp1, zp2)
    bad = lhs < rhs - 1e-8
    ctx.Assert(dreal.And(in_x0_cond(x01, x02), in_x_cond(z1, z2), in_x_cond(zp1, zp2), bad))
    m = ctx.CheckSat()
    if m is not None:
        ce_c3.append({
            'kind': 'c3',
            'x01': mid_float(m[x01]), 'x02': mid_float(m[x02]), 'z1': mid_float(m[z1]), 'z2': mid_float(m[z2]),
            'zp1': mid_float(m[zp1]), 'zp2': mid_float(m[zp2])
        })
    ce_all.extend(ce_c3)

    return ce_all


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ex2 closure-certificate synthesis")
    p.add_argument("--out", type=str, default="res_cc_ex2.json")
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    print_header("ex2", "CC", "closure_certificate", {"max_iter": args.max_iter, "grid_step": args.grid_step, "solver_synth": "z3", "solver_verify": "dreal"})
    result = synthesize(max_iter=args.max_iter, grid_step=args.grid_step, out_file=str(out_path))
    if result is None:
        result = {"success": False, "elapsed_sec": 0.0}
    if not out_path.exists():
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))
