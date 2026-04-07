import argparse
import json
import time
from pathlib import Path
import sys

from z3 import *

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

"""
ex3: two-room temperature model
This script synthesizes a transition-persistence certificate for transition q1 -> q1.
"""


def in_acc(q):
    return q == 1


def in_q0(q):
    return q == 0


def in_x_cond(x1, x2):
    return And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)


def in_x(x1, x2):
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x0_cond(x1, x2):
    return And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)


def in_x0(x1, x2):
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def q_trans_cond(x1, x2, qi, qj):
    if qi == 0 and qj == 1:
        return in_x0_cond(x1, x2)
    if qi == 0 and qj == 2:
        return Not(in_x0_cond(x1, x2))
    if (qi == 1 and qj == 1) or (qi == 2 and qj == 2):
        return True
    return False


def f_cond(x1, x2):
    alpha, theta, te, th, mu = 0.004, 0.01, 0, 40, 0.15

    def u(x):
        return 0.59 - 0.011 * x

    x1n = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te
    x2n = alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te
    return x1n, x2n


def f(x1, x2):
    alpha, theta, te, th, mu = 0.004, 0.01, 0, 40, 0.15

    def u(x):
        return 0.59 - 0.011 * x

    x1n = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te
    x2n = alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te
    return x1n, x2n


def label_value(x1, x2):
    cond1 = 21 <= x1 <= 24 and 21 <= x2 <= 24
    cond2 = 20 <= x1 <= 26 and 20 <= x2 <= 26
    if cond1 and cond2:
        return 3
    if cond1 and not cond2:
        return 2
    if not cond1 and cond2:
        return 1
    return 0


def b_cond(x1, x2):
    return And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)


def in_vf(x1, x2):
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def space_product(s1, s2):
    def conn(x1, x2):
        if type(x1) != list and type(x2) != list:
            return [x1, x2]
        if type(x1) == list and type(x2) != list:
            return x1 + [x2]
        if type(x1) != list and type(x2) == list:
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


def check_trans(x1, x2, qi, qj):
    lv = label_value(x1, x2)
    if lv >= 2 and qi == 0 and qj == 1:
        return True
    if lv < 2 and qi == 0 and qj == 2:
        return True
    if qi == 1 and qj == 1:
        return True
    if qi == 2 and qj == 2:
        return True
    return False


def step_sample(a, b, s):
    res = []
    for i in range(a * int(1 / s), b * int(1 / s) + 1):
        res.append(i * s)
    return res


def not_acc_cond(x1, x2, qi, qj):
    if qi != 1 or qj != 1:
        return True
    return Not(b_cond(x1, x2))


def t2float(v, precision=14):
    s = v.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)


def cegeis_synthesis(qi, qj, max_iter=1000):
    print(f"Synthesizing transition-persistence certificate for ({qi},{qj})...")
    found = False
    found_coeffs = None
    found_eps = None

    c = [Real(f'c{i}') for i in range(10)]
    epsilon = Real('epsilon')
    s = SolverFor("QF_NRA")

    x1_samples = step_sample(20, 34, 1)
    x2_samples = step_sample(20, 34, 1)
    samples = state_space_product(x1_samples, x2_samples)
    x01_samples = step_sample(21, 24, 0.1)
    x02_samples = step_sample(21, 24, 0.1)
    x0_samples = state_space_product(x01_samples, x02_samples)

    def bp(x1, x2):
        return simplify(
            c[0]
            + c[1] * If(in_x0_cond(x1, x2), 1, 0)
            + c[2] * If(b_cond(x1, x2), 1, 0)
            + c[3] * x1 * If(in_x0_cond(x1, x2), 1, 0)
            + c[4] * x2 * If(in_x0_cond(x1, x2), 1, 0)
            + c[5] * x1 * If(b_cond(x1, x2), 1, 0)
            + c[6] * x2 * If(b_cond(x1, x2), 1, 0)
            + c[7] * If(x1 > x2, x1, x2)
            + c[8] * x1 ** 2
            + c[9] * x2 ** 2
        )

    for x1, x2 in x0_samples:
        if check_trans(x1, x2, qi, qj):
            s.add(bp(x1, x2) >= 0)

    for x1, x2 in samples:
        if check_trans(x1, x2, qi, qj):
            x1n, x2n = f(x1, x2)
            if in_x(x1n, x2n):
                s.add(Implies(bp(x1, x2) >= 0, bp(x1n, x2n) >= 0))
                if qi == 1 and qj == 1 and in_vf(x1, x2):
                    s.add(bp(x1, x2) >= bp(x1n, x2n) + epsilon)
                else:
                    s.add(bp(x1, x2) >= bp(x1n, x2n))
    s.add(epsilon > 0)

    ce_solver = SolverFor("QF_NRA")
    x1_ce = Real('x1_ce')
    x2_ce = Real('x2_ce')
    x1_next_ce, x2_next_ce = f_cond(x1_ce, x2_ce)

    it = 0
    while s.check() == sat and it <= max_iter:
        ce_flag = False
        m = s.model()
        eps_val = t2float(m[epsilon])
        print(f"\n#{it + 1} candidate:")
        print(f"epsilon={eps_val}")
        for idx, item in enumerate(c):
            print(f"c{idx}={t2float(m[item])}")

        def bp_c(x1p, x2p):
            c_c = [m[item] for item in c]
            return simplify(
                c_c[0]
                + c_c[1] * If(in_x0_cond(x1p, x2p), 1, 0)
                + c_c[2] * If(b_cond(x1p, x2p), 1, 0)
                + c_c[3] * x1p * If(in_x0_cond(x1p, x2p), 1, 0)
                + c_c[4] * x2p * If(in_x0_cond(x1p, x2p), 1, 0)
                + c_c[5] * x1p * If(b_cond(x1p, x2p), 1, 0)
                + c_c[6] * x2p * If(b_cond(x1p, x2p), 1, 0)
                + c_c[7] * If(x1p > x2p, x1p, x2p)
                + c_c[8] * x1p ** 2
                + c_c[9] * x2p ** 2
            )

        def bp_num(x1p, x2p):
            c_c = [t2float(m[item]) for item in c]
            return (
                c_c[0]
                + c_c[1] * in_x0(x1p, x2p)
                + c_c[2] * in_vf(x1p, x2p)
                + c_c[3] * x1p * in_x0(x1p, x2p)
                + c_c[4] * x2p * in_x0(x1p, x2p)
                + c_c[5] * x1p * in_vf(x1p, x2p)
                + c_c[6] * x2p * in_vf(x1p, x2p)
                + c_c[7] * max(x1p, x2p)
                + c_c[8] * x1p ** 2
                + c_c[9] * x2p ** 2
            )

        # (1) Initial non-negativity
        ce_solver.push()
        ce_solver.add(q_trans_cond(x1_ce, x2_ce, qi, qj), in_x0_cond(x1_ce, x2_ce), bp_c(x1_ce, x2_ce) < 0)
        if ce_solver.check() == sat:
            print("Counterexample: initial non-negativity violated.")
            ce_flag = True
            ce_m = ce_solver.model()
            x1m, x2m = t2float(ce_m[x1_ce]), t2float(ce_m[x2_ce])
            print(f"x1={x1m}, x2={x2m}, B={bp_num(x1m, x2m)}")
            s.add(bp(x1m, x2m) >= 0)
        else:
            print("Initial non-negativity check passed.")
        ce_solver.pop()

        # (2) Transition non-negativity preservation
        ce_solver.push()
        ce_solver.add(
            q_trans_cond(x1_ce, x2_ce, qi, qj),
            in_x_cond(x1_ce, x2_ce),
            in_x_cond(x1_next_ce, x2_next_ce),
            not_acc_cond(x1_ce, x2_ce, qi, qj),
            bp_c(x1_ce, x2_ce) >= 0,
            bp_c(x1_next_ce, x2_next_ce) < 0,
        )
        if ce_solver.check() == sat:
            print("Counterexample: transition non-negativity preservation violated.")
            ce_flag = True
            ce_m = ce_solver.model()
            x1m, x2m = t2float(ce_m[x1_ce]), t2float(ce_m[x2_ce])
            x1n, x2n = f(x1m, x2m)
            print(f"({x1m}, {x2m}) -> ({x1n}, {x2n}), B={bp_num(x1m, x2m)}, B'={bp_num(x1n, x2n)}")
            s.add(Implies(bp(x1m, x2m) >= 0, bp(x1n, x2n) >= 0))
        else:
            print("Transition non-negativity preservation check passed.")
        ce_solver.pop()

        # (3) Non-increasing on non-accepting transitions
        ce_solver.push()
        ce_solver.add(
            q_trans_cond(x1_ce, x2_ce, qi, qj),
            in_x_cond(x1_ce, x2_ce),
            in_x_cond(x1_next_ce, x2_next_ce),
            not_acc_cond(x1_ce, x2_ce, qi, qj),
            bp_c(x1_ce, x2_ce) < bp_c(x1_next_ce, x2_next_ce),
        )
        if ce_solver.check() == sat:
            print("Counterexample: non-increasing property violated.")
            ce_flag = True
            ce_m = ce_solver.model()
            x1m, x2m = t2float(ce_m[x1_ce]), t2float(ce_m[x2_ce])
            x1n, x2n = f(x1m, x2m)
            print(f"({x1m}, {x2m}) -> ({x1n}, {x2n}), B={bp_num(x1m, x2m)}, B'={bp_num(x1n, x2n)}")
            s.add(bp(x1m, x2m) >= bp(x1n, x2n))
        else:
            print("Non-increasing check passed.")
        ce_solver.pop()

        # (4) Strict decrease on accepting transitions
        ce_solver.push()
        ce_solver.add(
            q_trans_cond(x1_ce, x2_ce, qi, qj),
            in_x_cond(x1_ce, x2_ce),
            in_x_cond(x1_next_ce, x2_next_ce),
            Not(not_acc_cond(x1_ce, x2_ce, qi, qj)),
            bp_c(x1_ce, x2_ce) < bp_c(x1_next_ce, x2_next_ce) + m[epsilon],
        )
        if ce_solver.check() == sat:
            print("Counterexample: strict decrease property violated.")
            ce_flag = True
            ce_m = ce_solver.model()
            x1m, x2m = t2float(ce_m[x1_ce]), t2float(ce_m[x2_ce])
            x1n, x2n = f(x1m, x2m)
            print(f"({x1m}, {x2m}) -> ({x1n}, {x2n}), B={bp_num(x1m, x2m)}, B'+eps={bp_num(x1n, x2n) + eps_val}")
            s.add(bp(ce_m[x1_ce], ce_m[x2_ce]) >= bp(*f_cond(ce_m[x1_ce], ce_m[x2_ce])) + epsilon)
        else:
            print("Strict decrease check passed.")
        ce_solver.pop()

        if not ce_flag:
            found = True
            found_coeffs = [t2float(m[item]) for item in c]
            found_eps = eps_val
            print("Certificate found.")
            break

        it += 1

    if it > max_iter:
        print("Exceeded maximum iterations.")
    elif not found:
        print("Unable to synthesize candidate certificate.")

    return {
        "success": found,
        "iterations": it,
        "max_iterations": max_iter,
        "epsilon": found_eps,
        "coefficients": found_coeffs,
    }


def main():
    parser = argparse.ArgumentParser(description="ex3 PT synthesis")
    parser.add_argument("--qi", type=int, default=1, help="source automaton state")
    parser.add_argument("--qj", type=int, default=1, help="target automaton state")
    parser.add_argument("--out", type=str, default="res_pt_ex3.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--grid-step", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--epochs", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--lr", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    args = parser.parse_args()

    print_header(
        "ex3",
        "PT",
        "transition_persistence",
        {"solver_synth": "z3", "solver_verify": "z3", "qi": args.qi, "qj": args.qj, "max_iter": args.max_iter},
    )
    start = time.time()
    result = cegeis_synthesis(args.qi, args.qj, args.max_iter)
    result["elapsed_sec"] = time.time() - start
    result["example"] = "ex3"
    result["method"] = "PT"
    result["certificate_type"] = "transition_persistence"
    result["solver"] = {"synth": "z3", "verify": "z3"}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), result["elapsed_sec"], str(out_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] ex3/PT failed: {e}")
        raise
