import argparse
import json
import math
import time
from pathlib import Path
import sys

import dreal
import z3

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

"""
ex1: 1D Kuramoto-like system
LTL target: G !Xu
Negation automaton target: F Xu
This script synthesizes a transition-persistence certificate for transition q1 -> q0.
"""

TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926


def in_x_cond(x):
    return dreal.And(x >= 0, x <= 2 * PI)


def in_x0_cond(x):
    return dreal.And(x >= 4 * PI / 9, x <= 5 * PI / 9)


def in_unsafe_cond(x):
    return dreal.And(x >= 7 * PI / 9, x <= 8 * PI / 9)


def in_unsafe(x):
    return 7 * PI / 9 <= x <= 8 * PI / 9


def f_t(x):
    return x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x ** 2 + 1.69


def f_m(x):
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69


def q_trans(q):
    if q == 1:
        return [0, 1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid q: {q}")


def delta(x, q):
    if q == 1:
        return [0] if in_unsafe(x) else [1]
    return [0]


def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res


def t2float(v, precision=14):
    s = v.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)


def space_product(s1, s2):
    def conn(x1, x2):
        if not isinstance(x1, list) and not isinstance(x2, list):
            return [x1, x2]
        if isinstance(x1, list) and not isinstance(x2, list):
            return x1 + [x2]
        if not isinstance(x1, list) and isinstance(x2, list):
            return [x1] + x2
        return x1 + x2

    if s1 == [] or s2 == []:
        return []

    return [conn(x1, x2) for x1 in s1 for x2 in s2]


def state_space_product(s1, *args):
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res


def reachability_10(max_iter=1000, grid_step=0.01, dreal_precision=1e-4, z3_timeout_ms=0):
    print("Synthesizing transition-persistence certificate for ex1 (q1->q0)...")
    cc_flag = False
    found_coeffs = None

    c = [z3.Real(f'c{i}') for i in range(9)]
    s = z3.SolverFor("QF_NRA")
    if z3_timeout_ms and z3_timeout_ms > 0:
        s.set("timeout", z3_timeout_ms)

    x_samples = step_sample(0, 2 * PI, grid_step)
    q_samples = [0, 1]
    y_samples = state_space_product(x_samples, q_samples)

    x0_samples = step_sample(4 * PI / 9, 5 * PI / 9, grid_step)
    y0_samples = state_space_product(x0_samples, [1])
    yu_samples = state_space_product(x_samples, [0])

    print(f"  |D0|={len(y0_samples)}, |Du|={len(yu_samples)}, |Dx|={len(y_samples)}")

    def bp_t(x, q):
        # B(x, q) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*q + c6*q^2 + c7*q^3 + c8*x*q
        res = (
            c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
            + c[5] * q + c[6] * q ** 2 + c[7] * q ** 3 + c[8] * x * q
        )
        return z3.simplify(res)

    for x0, q0 in y0_samples:
        s.add(bp_t(x0, q0) >= 0)
    for x, qacc in yu_samples:
        s.add(bp_t(x, qacc) < 0)
    for x, q in y_samples:
        xp = f_m(x)
        for qp in delta(x, q):
            s.add(z3.Implies(bp_t(x, q) >= 0, bp_t(xp, qp) >= 0))

    it = 0

    ce_solver = dreal.Context()
    ce_solver.config.precision = dreal_precision
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x_ce = dreal.Variable('x_ce')
    ce_solver.DeclareVariable(x_ce, 0, 2 * PI)

    while s.check() == z3.sat and it < max_iter:
        ce_flag = False
        m = s.model()
        c_m = [t2float(m[i]) for i in c]

        print(f"\nIteration {it + 1}")
        for i, cc in enumerate([m[item] for item in c]):
            print(f"c{i}={t2float(cc)}")

        def bp_c(x, q):
            return (
                c_m[0] + c_m[1] * x + c_m[2] * x ** 2 + c_m[3] * x ** 3 + c_m[4] * x ** 4
                + c_m[5] * q + c_m[6] * q ** 2 + c_m[7] * q ** 3 + c_m[8] * x * q
            )

        # Initial non-negativity
        ce_solver.Push(2)
        ce_solver.Assert(in_x0_cond(x_ce))
        ce_solver.Assert(bp_c(x_ce, 1) < 0)
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("Counterexample: initial non-negativity violated.")
            x_m = ce_model[x_ce].mid()
            print(f"x={x_m}, B={bp_c(x_m, 1)}")
            ce_flag = True
            s.add(bp_t(x_m, 1) >= 0)
        else:
            print("Initial non-negativity check passed.")
        ce_solver.Pop(2)

        # Unsafe negativity
        ce_solver.Push(2)
        ce_solver.Assert(in_x_cond(x_ce))
        ce_solver.Assert(bp_c(x_ce, 0) >= 0)
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("Counterexample: unsafe negativity violated.")
            x_m = ce_model[x_ce].mid()
            print(f"x={x_m}, B={bp_c(x_m, 0)}")
            ce_flag = True
            s.add(bp_t(x_m, 0) < 0)
        else:
            print("Unsafe negativity check passed.")
        ce_solver.Pop(2)

        # Transition preservation
        tnn_flag = False
        ce_solver.Push(3)
        ce_solver.Assert(in_x_cond(x_ce))
        ce_solver.Assert(in_x_cond(f_t(x_ce)))
        ce_solver.Assert(bp_c(x_ce, 1) >= 0)
        for qp in q_trans(1):
            ce_solver.Push(2)
            if qp == 0:
                ce_solver.Assert(in_unsafe_cond(x_ce))
                ce_solver.Assert(bp_c(f_t(x_ce), qp) < 0)
            elif qp == 1:
                ce_solver.Assert(dreal.Not(in_unsafe_cond(x_ce)))
                ce_solver.Assert(bp_c(f_t(x_ce), qp) < 0)
            else:
                raise ValueError(f"invalid qp: {qp}")

            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print("Counterexample: transition preservation violated (q=1).")
                ce_flag = True
                tnn_flag = True
                x_m = ce_model[x_ce].mid()
                xp_m = f_m(x_m)
                print(f"B(x,q)={bp_c(x_m, 1)}, B(x',q')={bp_c(xp_m, qp)}")
                s.add(z3.Implies(bp_t(x_m, 1) >= 0, bp_t(xp_m, qp) >= 0))
            ce_solver.Pop(2)
        ce_solver.Pop(3)

        ce_solver.Push(3)
        ce_solver.Assert(in_x_cond(x_ce))
        ce_solver.Assert(in_x_cond(f_t(x_ce)))
        ce_solver.Assert(bp_c(x_ce, 0) >= 0)
        for qp in q_trans(0):
            ce_solver.Push(1)
            if qp == 0:
                ce_solver.Assert(bp_c(f_t(x_ce), qp) < 0)
            else:
                raise ValueError(f"invalid qp: {qp}")

            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print("Counterexample: transition preservation violated (q=0).")
                ce_flag = True
                tnn_flag = True
                x_m = ce_model[x_ce].mid()
                xp_m = f_m(x_m)
                print(f"B(x,q)={bp_c(x_m, 0)}, B(x',q')={bp_c(xp_m, qp)}")
                s.add(z3.Implies(bp_t(x_m, 0) >= 0, bp_t(xp_m, qp) >= 0))
            ce_solver.Pop(1)
        ce_solver.Pop(3)

        if not tnn_flag:
            print("Transition preservation checks passed.")

        if not ce_flag:
            print("Certificate found.")
            cc_flag = True
            found_coeffs = [t2float(m[item]) for item in c]
            for i, cc in enumerate([m[item] for item in c]):
                print(f"c{i}={cc}")
            break

        it += 1

    if it >= max_iter:
        print("Reached maximum iterations without convergence.")
    elif not cc_flag:
        print("Unable to synthesize a valid certificate.")
    else:
        print(f"Converged in {it} iterations.")

    return {
        "success": cc_flag,
        "iterations": it,
        "max_iterations": max_iter,
        "coefficients": found_coeffs,
    }


def main():
    parser = argparse.ArgumentParser(description="ex1 PT synthesis")
    parser.add_argument("--out", type=str, default="res_pt_ex1.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    parser.add_argument("--z3-timeout-ms", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = parser.parse_args()

    print_header(
        "ex1",
        "PT",
        "transition_safety",
        {
            "solver_synth": "z3",
            "solver_verify": "dreal",
            "max_iter": args.max_iter,
            "grid_step": args.grid_step,
            "dreal_precision": args.dreal_precision,
            "z3_timeout_ms": args.z3_timeout_ms,
        },
    )
    start_time = time.time()
    result = reachability_10(
        max_iter=args.max_iter,
        grid_step=args.grid_step,
        dreal_precision=args.dreal_precision,
        z3_timeout_ms=args.z3_timeout_ms,
    )
    result["elapsed_sec"] = time.time() - start_time
    result["example"] = "ex1"
    result["method"] = "PT"
    result["certificate_type"] = "transition_safety"
    result["solver"] = {"synth": "z3", "verify": "dreal"}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), result["elapsed_sec"], str(out_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] ex1/PT failed: {e}")
        raise
