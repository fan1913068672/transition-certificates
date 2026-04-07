import argparse
import json
import random
import time
from pathlib import Path
import sys

import z3

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result


# S-procedure style strengthening constants (Closure Certificates paper).
TAU1 = 1.0
TAU2 = 1.0
TAU3 = 0.0


def f(x1, x2):
    alpha, theta, mu, th, te = 0.004, 0.01, 0.15, 40.0, 0.0

    def u(x):
        return 0.59 - 0.011 * x

    return (
        (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te,
    )


def in_x(x1, x2):
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x0(x1, x2):
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def in_vf(x1, x2):
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def in_x_cond(x1, x2):
    return z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)


def in_x0_cond(x1, x2):
    return z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)


def in_vf_cond(x1, x2):
    return z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)


def labels(x1, x2):
    return int(in_x0(x1, x2)), int(in_vf(x1, x2))


def delta_main_pt(q, x1, x2):
    a0, _ = labels(x1, x2)
    if q == 0:
        return [1] if a0 == 1 else [2]
    if q == 1:
        return [1]
    return [2]


def delta_closure_paper(q, x1, x2):
    a0, a1 = labels(x1, x2)
    if q == 0:
        if (a0, a1) == (1, 1):
            return [2]
        if (a0, a1) == (0, 1):
            return [1, 3]
        if (a0, a1) == (1, 0):
            return [1]
        return [3]
    if q == 1:
        return [2] if a1 == 1 else [1]
    if q == 2:
        return [2]
    return [3]


def delta_branch_cond(mode, i, ip, x1, x2):
    a0 = in_x0_cond(x1, x2)
    a1 = in_vf_cond(x1, x2)
    if mode == "main":
        if i == 0 and ip == 1:
            return a0
        if i == 0 and ip == 2:
            return z3.Not(a0)
        if i == 1 and ip == 1:
            return z3.BoolVal(True)
        if i == 2 and ip == 2:
            return z3.BoolVal(True)
        return z3.BoolVal(False)

    # closure
    if i == 0 and ip == 2:
        return z3.And(a0, a1)
    if i == 0 and ip == 1:
        return z3.Or(z3.And(a0, z3.Not(a1)), z3.And(z3.Not(a0), a1))
    if i == 0 and ip == 3:
        return z3.Or(z3.And(z3.Not(a0), z3.Not(a1)), z3.And(z3.Not(a0), a1))
    if i == 1 and ip == 2:
        return a1
    if i == 1 and ip == 1:
        return z3.Not(a1)
    if i == 2 and ip == 2:
        return z3.BoolVal(True)
    if i == 3 and ip == 3:
        return z3.BoolVal(True)
    return z3.BoolVal(False)


class Template:
    def __init__(self, q_states):
        self.q_states = q_states
        self.c = {(i, j): [z3.Real(f"c_{i}_{j}_{k}") for k in range(11)] for i in q_states for j in q_states}

    def T_z3(self, i, j, x1, x2, y1, y2):
        c = self.c[(i, j)]
        return (
            c[0]
            + c[1] * x1
            + c[2] * x2
            + c[3] * y1
            + c[4] * y2
            + c[5] * z3.If(x1 > x2, x1, x2)
            + c[6] * z3.If(y1 > y2, y1, y2)
            + c[7] * x1 * x1
            + c[8] * x2 * x2
            + c[9] * y1 * y1
            + c[10] * y2 * y2
        )


def grid(a, b, step):
    vals = []
    v = a
    while v <= b + 1e-9:
        vals.append(round(v, 8))
        v += step
    return vals


def sample_points(grid_step=1.0):
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0")
    xs = grid(20, 34, grid_step)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    vf = [(x1, x2) for x1, x2 in pts if in_vf(x1, x2)]
    return pts, x0, vf


def t2float(v, precision=16):
    vv = v
    if vv is None:
        return 0.0
    if z3.is_rational_value(vv):
        return float(vv.numerator_as_long()) / float(vv.denominator_as_long())
    s = vv.as_decimal(precision)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def find_counterexamples(mode, tpl, coeffs, eps_val, q_states, q0, qacc, max_per_kind=2):
    ce_list = []

    def tnum(i, j, x1, x2, y1, y2):
        c = coeffs[(i, j)]
        return (
            c[0]
            + c[1] * x1
            + c[2] * x2
            + c[3] * y1
            + c[4] * y2
            + c[5] * max(x1, x2)
            + c[6] * max(y1, y2)
            + c[7] * x1 * x1
            + c[8] * x2 * x2
            + c[9] * y1 * y1
            + c[10] * y2 * y2
        )

    # C1
    c1_cnt = 0
    for i in q_states:
        for ip in q_states:
            if c1_cnt >= max_per_kind:
                break
            s = z3.SolverFor("QF_NRA")
            x1, x2 = z3.Reals("x1 x2")
            s.add(in_x_cond(x1, x2))
            s.add(delta_branch_cond(mode, i, ip, x1, x2))
            x1n, x2n = f(x1, x2)
            s.add(in_x_cond(x1n, x2n))
            s.add(tpl.T_z3(i, ip, x1, x2, x1n, x2n) < 0)
            if s.check() == z3.sat:
                m = s.model()
                ce_list.append(
                    {
                        "kind": "c1",
                        "i": i,
                        "ip": ip,
                        "x1": t2float(m[x1]),
                        "x2": t2float(m[x2]),
                    }
                )
                c1_cnt += 1

    # C2 (strengthened Eq.(24)-style)
    c2_cnt = 0
    for i in q_states:
        for ip in q_states:
            for j in q_states:
                if c2_cnt >= max_per_kind:
                    break
                s = z3.SolverFor("QF_NRA")
                x1, x2, y1, y2 = z3.Reals("x1 x2 y1 y2")
                s.add(in_x_cond(x1, x2), in_x_cond(y1, y2))
                s.add(delta_branch_cond(mode, i, ip, x1, x2))
                x1n, x2n = f(x1, x2)
                s.add(in_x_cond(x1n, x2n))
                lhs = tpl.T_z3(i, j, x1, x2, y1, y2)
                rhs = TAU1 * tpl.T_z3(ip, j, x1n, x2n, y1, y2)
                s.add(lhs < rhs)
                if s.check() == z3.sat:
                    m = s.model()
                    ce_list.append(
                        {
                            "kind": "c2",
                            "i": i,
                            "ip": ip,
                            "j": j,
                            "x1": t2float(m[x1]),
                            "x2": t2float(m[x2]),
                            "y1": t2float(m[y1]),
                            "y2": t2float(m[y2]),
                        }
                    )
                    c2_cnt += 1

    # C3 (strengthened Eq.(25)-style)
    s = z3.SolverFor("QF_NRA")
    x01, x02, z1, z2, zp1, zp2 = z3.Reals("x01 x02 z1 z2 zp1 zp2")
    s.add(in_x0_cond(x01, x02), in_vf_cond(z1, z2), in_vf_cond(zp1, zp2))
    lhs = tpl.T_z3(q0, qacc, x01, x02, z1, z2) - eps_val - tpl.T_z3(q0, qacc, x01, x02, zp1, zp2)
    rhs = TAU2 * tpl.T_z3(q0, qacc, x01, x02, z1, z2) + TAU3 * tpl.T_z3(qacc, qacc, z1, z2, zp1, zp2)
    s.add(lhs < rhs)
    if s.check() == z3.sat:
        m = s.model()
        ce_list.append(
            {
                "kind": "c3",
                "x01": t2float(m[x01]),
                "x02": t2float(m[x02]),
                "z1": t2float(m[z1]),
                "z2": t2float(m[z2]),
                "zp1": t2float(m[zp1]),
                "zp2": t2float(m[zp2]),
            }
        )

    return ce_list


def solve(mode="main", max_iter=200, grid_step=1.0, out_file="res_cc_ex3.json"):
    start = time.time()
    pts, x0_pts, vf_pts = sample_points(grid_step=grid_step)
    print(f"Dataset sizes: |X|={len(pts)}, |X0|={len(x0_pts)}, |XVF|={len(vf_pts)}, grid_step={grid_step}")

    if mode == "closure":
        q_states = [0, 1, 2, 3]
        delta = delta_closure_paper
        q0, qacc = 0, 2
    else:
        q_states = [0, 1, 2]
        delta = delta_main_pt
        q0, qacc = 0, 1

    tpl = Template(q_states)
    s = z3.SolverFor("QF_NRA")

    for coeffs in tpl.c.values():
        for ci in coeffs:
            s.add(ci >= -20, ci <= 20)

    eps = z3.Real("eps")
    s.add(eps > 0, eps <= 5)

    # Initial sampled seeds
    for x1, x2 in random.sample(pts, min(120, len(pts))):
        x1n, x2n = f(x1, x2)
        if not in_x(x1n, x2n):
            continue
        for i in q_states:
            for ip in delta(i, x1, x2):
                s.add(tpl.T_z3(i, ip, x1, x2, x1n, x2n) >= 0)

    z_pool = random.sample(pts, min(80, len(pts)))
    for x1, x2 in random.sample(pts, min(80, len(pts))):
        x1n, x2n = f(x1, x2)
        if not in_x(x1n, x2n):
            continue
        for i in q_states:
            for ip in delta(i, x1, x2):
                for y1, y2 in z_pool[:20]:
                    for j in q_states:
                        s.add(tpl.T_z3(i, j, x1, x2, y1, y2) >= TAU1 * tpl.T_z3(ip, j, x1n, x2n, y1, y2))

    for x01, x02 in random.sample(x0_pts, min(20, len(x0_pts))):
        for z1, z2 in random.sample(vf_pts, min(30, len(vf_pts))):
            zp1, zp2 = f(z1, z2)
            if not in_x(zp1, zp2):
                continue
            lhs = tpl.T_z3(q0, qacc, x01, x02, z1, z2) - eps - tpl.T_z3(q0, qacc, x01, x02, zp1, zp2)
            rhs = TAU2 * tpl.T_z3(q0, qacc, x01, x02, z1, z2) + TAU3 * tpl.T_z3(qacc, qacc, z1, z2, zp1, zp2)
            s.add(lhs >= rhs)

    for it in range(max_iter):
        print(f"Iteration {it+1}")
        if s.check() != z3.sat:
            print("unsat")
            return {"success": False, "mode": mode, "iterations": it + 1, "elapsed_sec": time.time() - start}

        m = s.model()
        eps_val = t2float(m[eps])
        coeff_vals = {(i, j): [t2float(m[c]) for c in tpl.c[(i, j)]] for i in q_states for j in q_states}
        print(f"candidate eps={eps_val:.6f}")

        ce_list = find_counterexamples(mode, tpl, coeff_vals, eps_val, q_states, q0, qacc, max_per_kind=2)
        if len(ce_list) == 0:
            out = {"success": True, "mode": mode, "eps": eps_val, "iterations": it + 1, "elapsed_sec": time.time() - start}
            for (i, j), vv in coeff_vals.items():
                out[f"T_{i}_{j}"] = vv
            Path(out_file).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print("saved", out_file)
            return out

        c1c = sum(1 for ce in ce_list if ce["kind"] == "c1")
        c2c = sum(1 for ce in ce_list if ce["kind"] == "c2")
        c3c = sum(1 for ce in ce_list if ce["kind"] == "c3")
        print(f"added CE batch: c1={c1c}, c2={c2c}, c3={c3c}, total={len(ce_list)}")

        for ce in ce_list:
            k = ce["kind"]
            if k == "c1":
                x1, x2, i, ip = ce["x1"], ce["x2"], ce["i"], ce["ip"]
                x1n, x2n = f(x1, x2)
                s.add(tpl.T_z3(i, ip, x1, x2, x1n, x2n) >= 0)
            elif k == "c2":
                x1, x2, y1, y2, i, ip, j = ce["x1"], ce["x2"], ce["y1"], ce["y2"], ce["i"], ce["ip"], ce["j"]
                x1n, x2n = f(x1, x2)
                s.add(tpl.T_z3(i, j, x1, x2, y1, y2) >= TAU1 * tpl.T_z3(ip, j, x1n, x2n, y1, y2))
            else:
                x01, x02, z1, z2, zp1, zp2 = ce["x01"], ce["x02"], ce["z1"], ce["z2"], ce["zp1"], ce["zp2"]
                lhs = tpl.T_z3(q0, qacc, x01, x02, z1, z2) - eps - tpl.T_z3(q0, qacc, x01, x02, zp1, zp2)
                rhs = TAU2 * tpl.T_z3(q0, qacc, x01, x02, z1, z2) + TAU3 * tpl.T_z3(qacc, qacc, z1, z2, zp1, zp2)
                s.add(lhs >= rhs)

    print("max_iter reached")
    return {"success": False, "mode": mode, "iterations": max_iter, "elapsed_sec": time.time() - start}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ex3 closure-certificate synthesis")
    p.add_argument("--mode", choices=["main", "closure"], default="main")
    p.add_argument("--out", default="res_cc_ex3.json")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--grid-step", type=float, default=1.0)
    p.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    p.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    p.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    print_header(
        "ex3",
        "CC",
        "closure_certificate",
        {"mode": args.mode, "max_iter": args.max_iter, "grid_step": args.grid_step, "solver_synth": "z3", "solver_verify": "z3"},
    )
    result = solve(mode=args.mode, max_iter=args.max_iter, grid_step=args.grid_step, out_file=str(out_path))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))
