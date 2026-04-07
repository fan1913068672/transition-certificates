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
ex2 PT (strictly ex1/PT-style CEGIS structure, 2D adaptation)
Certificate type: transition safety for q=0 (accepting-start state unreachable).
"""

PI = 3.1415926
TS = 0.1
OMEGA = 0.01
K = 0.0006


def in_x(x1, x2):
    return x1 >= 0 and x1 <= 8 * PI / 9 and x2 >= 0 and x2 <= 8 * PI / 9


def in_x_cond(x1, x2):
    return dreal.And(dreal.And(x1 >= 0, x1 <= 8 * PI / 9), dreal.And(x2 >= 0, x2 <= 8 * PI / 9))


def in_x0_cond(x1, x2):
    return dreal.And(dreal.And(x1 >= 0, x1 <= PI / 9), dreal.And(x2 >= 0, x2 <= PI / 9))


def in_unsafe(x1, x2):
    return (x1 >= 5 * PI / 6 and x1 <= 8 * PI / 9) or (x2 >= 5 * PI / 6 and x2 <= 8 * PI / 9)


def in_unsafe_cond(x1, x2):
    return dreal.Or(
        dreal.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
        dreal.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
    )


def f_t(x1, x2):
    x1p = x1 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p


def f_m(x1, x2):
    x1p = x1 + TS * OMEGA + 1.69 + TS * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + TS * OMEGA + 1.69 + TS * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p


def q_trans(q):
    if q == 1:
        return [0, 1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid q: {q}")


def delta(x1, x2, q):
    if q == 1:
        return [0] if in_unsafe(x1, x2) else [1]
    return [0]


def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res


def t2float(v, precision=14):
    if v is None:
        return 0.0
    s = v.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)


def state_space_product(s1, s2, s3):
    return [(x1, x2, q) for x1 in s1 for x2 in s2 for q in s3]


def synthesize(
    max_iter=1000,
    grid_step=0.1,
    dreal_precision=1e-4,
    z3_timeout_ms=0,
):
    print("Synthesizing transition-safety certificate for q=0 (ex2)...")
    cc_flag = False
    found_coeffs = None

    c = [z3.Real(f'c{i}') for i in range(7)]
    # Candidate synthesis is solved by Z3 on sampled constraints.
    s = z3.SolverFor("QF_NRA")
    if z3_timeout_ms and z3_timeout_ms > 0:
        s.set("timeout", z3_timeout_ms)

    x1_samples = step_sample(0, 8 * PI / 9, grid_step)
    x2_samples = step_sample(0, 8 * PI / 9, grid_step)
    y_samples = state_space_product(x1_samples, x2_samples, [0, 1])
    y0_samples = state_space_product(step_sample(0, PI / 9, grid_step), step_sample(0, PI / 9, grid_step), [1])
    yu_samples = state_space_product(x1_samples, x2_samples, [0])  # unsafe product states

    print(f"Dataset sizes: |D0|={len(y0_samples)}, |Du|={len(yu_samples)}, |Dx|={len(y_samples)}")

    def bp_t(x1, x2, q):
        # User-requested template:
        # B = c0 + c1*x1 + c2*x2 + c3*q + c4*I_Xu(x) + c5*x1*q + c6*x2*q
        iu = 1.0 if in_unsafe(float(x1), float(x2)) else 0.0
        res = (
            c[0]
            + c[1] * x1
            + c[2] * x2
            + c[3] * q
            + c[4] * iu
            + c[5] * x1 * q
            + c[6] * x2 * q
        )
        return z3.simplify(res)

    for x1, x2, q0 in y0_samples:
        s.add(bp_t(x1, x2, q0) >= 0)
    for x1, x2, qacc in yu_samples:
        s.add(bp_t(x1, x2, qacc) < 0)
    for x1, x2, q in y_samples:
        x1p, x2p = f_m(x1, x2)
        for qp in delta(x1, x2, q):
            s.add(z3.Implies(bp_t(x1, x2, q) >= 0, bp_t(x1p, x2p, qp) >= 0))

    it = 0
    ce_solver = dreal.Context()
    ce_solver.config.precision = dreal_precision
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x1_ce = dreal.Variable('x1_ce')
    x2_ce = dreal.Variable('x2_ce')
    ce_solver.DeclareVariable(x1_ce, 0, 8 * PI / 9)
    ce_solver.DeclareVariable(x2_ce, 0, 8 * PI / 9)

    chk = None
    while it < max_iter:
        if z3_timeout_ms and z3_timeout_ms > 0:
            print(f"\n[Z3] checking candidate at iteration {it + 1} (timeout={z3_timeout_ms} ms)...")
        else:
            print(f"\n[Z3] checking candidate at iteration {it + 1} (no timeout)...")
        chk = s.check()
        print(f"[Z3] status: {chk}")
        if chk != z3.sat:
            break
        ce_flag = False
        m = s.model()
        # Some coefficients may be omitted in model printing; complete them as 0.
        c_m = [t2float(m.evaluate(ci, model_completion=True)) for ci in c]

        print(f"\nIteration {it + 1}")
        for i, cc in enumerate([m.evaluate(item, model_completion=True) for item in c]):
            print(f"c{i}={t2float(cc)}")

        def bp_c(x1, x2, q):
            iu = dreal.if_then_else(in_unsafe_cond(x1, x2), 1.0, 0.0)
            return (
                c_m[0]
                + c_m[1] * x1
                + c_m[2] * x2
                + c_m[3] * q
                + c_m[4] * iu
                + c_m[5] * x1 * q
                + c_m[6] * x2 * q
            )

        # Initial non-negativity
        ce_solver.Push(2)
        ce_solver.Assert(in_x0_cond(x1_ce, x2_ce))
        ce_solver.Assert(bp_c(x1_ce, x2_ce, 1) < 0)
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("Counterexample: initial non-negativity violated.")
            x1_m = ce_model[x1_ce].mid()
            x2_m = ce_model[x2_ce].mid()
            print(f"{x1_m}, {x2_m}, q=1, B={bp_c(x1_m, x2_m, 1)}")
            ce_flag = True
            s.add(bp_t(x1_m, x2_m, 1) >= 0)
        else:
            print("Initial non-negativity check passed.")
        ce_solver.Pop(2)

        # Unsafe negativity for q=0
        ce_solver.Push(2)
        ce_solver.Assert(in_x_cond(x1_ce, x2_ce))
        ce_solver.Assert(bp_c(x1_ce, x2_ce, 0) >= 0)
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("Counterexample: unsafe negativity violated.")
            x1_m = ce_model[x1_ce].mid()
            x2_m = ce_model[x2_ce].mid()
            print(f"{x1_m}, {x2_m}, q=0, B={bp_c(x1_m, x2_m, 0)}")
            ce_flag = True
            s.add(bp_t(x1_m, x2_m, 0) < 0)
        else:
            print("Unsafe negativity check passed.")
        ce_solver.Pop(2)

        # Transition preservation
        tnn_flag = False

        ce_solver.Push(3)
        ce_solver.Assert(in_x_cond(x1_ce, x2_ce))
        ce_solver.Assert(in_x_cond(*f_t(x1_ce, x2_ce)))
        ce_solver.Assert(bp_c(x1_ce, x2_ce, 1) >= 0)
        for qp in q_trans(1):
            ce_solver.Push(2)
            if qp == 0:
                ce_solver.Assert(in_unsafe_cond(x1_ce, x2_ce))
                ce_solver.Assert(bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            elif qp == 1:
                ce_solver.Assert(dreal.Not(in_unsafe_cond(x1_ce, x2_ce)))
                ce_solver.Assert(bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            else:
                raise ValueError(f"invalid qp: {qp}")

            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print(f"Counterexample: transition preservation violated from q=1 to q={qp}.")
                ce_flag = True
                tnn_flag = True
                x1_m = ce_model[x1_ce].mid()
                x2_m = ce_model[x2_ce].mid()
                x1_n, x2_n = f_m(x1_m, x2_m)
                print(f"({x1_m}, {x2_m}, q=1) -> ({x1_n}, {x2_n}, q={qp})")
                s.add(z3.Implies(bp_t(x1_m, x2_m, 1) >= 0, bp_t(x1_n, x2_n, qp) >= 0))
                ce_solver.Pop(2)
                break
            ce_solver.Pop(2)
        ce_solver.Pop(3)

        # ex1-style can keep checking all branches, but for ex2 we stop at first CE
        # to avoid long dReal stalls in the same iteration.
        if ce_flag:
            it += 1
            continue

        ce_solver.Push(3)
        ce_solver.Assert(in_x_cond(x1_ce, x2_ce))
        ce_solver.Assert(in_x_cond(*f_t(x1_ce, x2_ce)))
        ce_solver.Assert(bp_c(x1_ce, x2_ce, 0) >= 0)
        for qp in q_trans(0):
            ce_solver.Push(1)
            if qp == 0:
                ce_solver.Assert(bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            else:
                raise ValueError(f"invalid qp: {qp}")

            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print("Counterexample: transition preservation violated from q=0.")
                ce_flag = True
                tnn_flag = True
                x1_m = ce_model[x1_ce].mid()
                x2_m = ce_model[x2_ce].mid()
                x1_n, x2_n = f_m(x1_m, x2_m)
                print(f"({x1_m}, {x2_m}, q=0) -> ({x1_n}, {x2_n}, q={qp})")
                s.add(z3.Implies(bp_t(x1_m, x2_m, 0) >= 0, bp_t(x1_n, x2_n, qp) >= 0))
                ce_solver.Pop(1)
                break
            ce_solver.Pop(1)
        ce_solver.Pop(3)

        if ce_flag:
            it += 1
            continue

        if not tnn_flag:
            print("Transition preservation checks passed.")

        if not ce_flag:
            print("Certificate found.")
            cc_flag = True
            found_coeffs = [t2float(m.evaluate(item, model_completion=True)) for item in c]
            for i, cc in enumerate([m.evaluate(item, model_completion=True) for item in c]):
                print(f"c{i}={cc}")
            break

        it += 1

    if it >= max_iter:
        print("Reached maximum iterations without convergence.")
    elif chk == z3.unsat:
        print("Z3 returned UNSAT: no candidate satisfies current constraints.")
    elif chk == z3.unknown:
        print("Z3 returned UNKNOWN (likely timeout/complexity).")
    elif not cc_flag:
        print("Unable to synthesize a valid certificate.")
    else:
        print(f"Converged in {it} iterations.")

    return {
        "success": cc_flag,
        "certificate_type": "transition_safety",
        "target_q": 0,
        "iterations": it,
        "max_iterations": max_iter,
        "grid_step": grid_step,
        "dreal_precision": dreal_precision,
        "z3_timeout_ms": z3_timeout_ms,
        "coefficients": found_coeffs,
    }


def main():
    parser = argparse.ArgumentParser(description="ex2 PT (transition safety)")
    parser.add_argument("--out", type=str, default="res_pt_ex2.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    parser.add_argument("--z3-timeout-ms", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = parser.parse_args()

    print_header(
        "ex2",
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
    result = synthesize(
        args.max_iter,
        args.grid_step,
        args.dreal_precision,
        args.z3_timeout_ms,
    )
    result["elapsed_sec"] = time.time() - start_time
    result["example"] = "ex2"
    result["method"] = "PT"
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
        print(f"[ERROR] ex2/PT failed: {e}")
        raise
