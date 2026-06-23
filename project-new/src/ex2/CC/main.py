from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import z3
try:
    import dreal
except ModuleNotFoundError:  # dReal is only needed for the optional exact nonlinear backend.
    dreal = None

sys.path.append(str(Path(__file__).resolve().parents[2]))
from closure_smt import ClosureCase, synthesize_closure
from run_output_utils import print_header, print_result
from state_triplet_smt import Automaton, label

PI = 3.1415926
TS = 0.1
OMEGA = 0.01
K = 0.0006


def in_x(x1: float, x2: float) -> bool:
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x0(x1: float, x2: float) -> bool:
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


def in_unsafe(x1: float, x2: float) -> bool:
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def in_x_z3(x1, x2):
    return z3.And(x1 >= 0, x1 <= 8 * PI / 9, x2 >= 0, x2 <= 8 * PI / 9)


def in_x0_z3(x1, x2):
    return z3.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9)


def in_unsafe_z3(x1, x2):
    return z3.Or(z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9), z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9))


def in_x_dreal(x1, x2):
    return dreal.And(x1 >= 0, x1 <= 8 * PI / 9, x2 >= 0, x2 <= 8 * PI / 9)


def in_x0_dreal(x1, x2):
    return dreal.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9)


def in_unsafe_dreal(x1, x2):
    return dreal.Or(dreal.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9), dreal.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9))


def f_num(x1: float, x2: float):
    return (
        x1 + TS * OMEGA + 1.69 + TS * K * math.sin(x2 - x1) - 0.532 * TS * x1**2,
        x2 + TS * OMEGA + 1.69 + TS * K * math.sin(x1 - x2) - 0.532 * TS * x2**2,
    )


def f_z3(x1, x2):
    return (
        x1 + TS * OMEGA + 1.69 - 0.532 * TS * x1**2,
        x2 + TS * OMEGA + 1.69 - 0.532 * TS * x2**2,
    )


def f_interval_relation_z3(x, xn):
    """Sound polynomial over-approximation of the sin-coupled transition.

    Since |sin(.)| <= 1, the exact transition
        x1' = x1 + TS*OMEGA + 1.69 - 0.532*TS*x1^2 + TS*K*sin(x2-x1)
    is contained in the interval relation below.  Verifying the closure
    conditions for all xn in this relation is therefore stronger than verifying
    them for the exact Kuramoto step and remains a formal Z3 proof obligation.
    """
    x1, x2 = x
    xn1, xn2 = xn
    b1 = x1 + TS * OMEGA + 1.69 - 0.532 * TS * x1**2
    b2 = x2 + TS * OMEGA + 1.69 - 0.532 * TS * x2**2
    r = TS * K
    return z3.And(xn1 >= b1 - r, xn1 <= b1 + r, xn2 >= b2 - r, xn2 <= b2 + r)


def f_dreal(x1, x2):
    if dreal is None:
        raise RuntimeError("dReal is unavailable")
    return (
        x1 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x2 - x1) - 0.532 * TS * x1**2,
        x2 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x1 - x2) - 0.532 * TS * x2**2,
    )


def step_sample(a: float, b: float, s: float):
    vals = []
    n0 = int(round(a / s))
    n1 = int(round(b / s))
    for i in range(n0, n1 + 1):
        vals.append(round(i * s, 10))
    return vals


def label_of(x1: float, x2: float):
    if in_unsafe(x1, x2):
        return label("a")
    return label()


def build_case(dreal_precision: float = 1e-4) -> ClosureCase:
    automaton = Automaton(
        states=[0, 1],
        initial_states=[1],
        accepting_states=[0],
        transitions={
            (1, label()): [1],
            (1, label("a")): [0],
            (0, label()): [0],
            (0, label("a")): [0],
        },
    )
    # Around 400 grid points, matching the original CC case-study setup.
    xs = step_sample(0.0, 8 * PI / 9, 0.15)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0s = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    return ClosureCase(
        name="ex2",
        automaton=automaton,
        dim=2,
        q0=1,
        accepting_states=[0],
        sample_points=pts,
        x0_samples=x0s,
        domain_bounds=[(0.0, 8 * PI / 9), (0.0, 8 * PI / 9)],
        domain_member=lambda x1, x2: in_x(x1, x2),
        x0_member=lambda x1, x2: in_x0(x1, x2),
        next_num=lambda x1, x2: f_num(x1, x2),
        next_z3=lambda x1, x2: f_z3(x1, x2),
        next_dreal=(lambda x1, x2: f_dreal(x1, x2)) if dreal is not None else None,
        label_of=lambda x1, x2: label_of(x1, x2),
        label_cond_z3={
            label(): lambda x1, x2: z3.And(in_x_z3(x1, x2), z3.Not(in_unsafe_z3(x1, x2))),
            label("a"): lambda x1, x2: in_unsafe_z3(x1, x2),
        },
        label_cond_dreal=(
            {
                label(): lambda x1, x2: dreal.And(in_x_dreal(x1, x2), dreal.Not(in_unsafe_dreal(x1, x2))),
                label("a"): lambda x1, x2: in_unsafe_dreal(x1, x2),
            }
            if dreal is not None
            else None
        ),
        domain_cond_z3=lambda x1, x2: in_x_z3(x1, x2),
        domain_cond_dreal=(lambda x1, x2: in_x_dreal(x1, x2)) if dreal is not None else None,
        x0_cond_z3=lambda x1, x2: in_x0_z3(x1, x2),
        x0_cond_dreal=(lambda x1, x2: in_x0_dreal(x1, x2)) if dreal is not None else None,
        template_terms_num=[
            lambda x1, x2, y1, y2: 1.0,
            lambda x1, x2, y1, y2: y1 * (1.0 if in_x0(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: y2 * (1.0 if in_x0(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: y1 * (1.0 if in_unsafe(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: y2 * (1.0 if in_unsafe(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: y1,
            lambda x1, x2, y1, y2: y2,
        ],
        template_terms_z3=[
            lambda x1, x2, y1, y2: 1.0,
            lambda x1, x2, y1, y2: y1 * z3.If(in_x0_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y2 * z3.If(in_x0_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y1 * z3.If(in_unsafe_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y2 * z3.If(in_unsafe_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y1,
            lambda x1, x2, y1, y2: y2,
        ],
        template_terms_dreal=(
            [
                lambda x1, x2, y1, y2: 1.0,
                lambda x1, x2, y1, y2: y1 * dreal.if_then_else(in_x0_dreal(x1, x2), 1.0, 0.0),
                lambda x1, x2, y1, y2: y2 * dreal.if_then_else(in_x0_dreal(x1, x2), 1.0, 0.0),
                lambda x1, x2, y1, y2: y1 * dreal.if_then_else(in_unsafe_dreal(x1, x2), 1.0, 0.0),
                lambda x1, x2, y1, y2: y2 * dreal.if_then_else(in_unsafe_dreal(x1, x2), 1.0, 0.0),
                lambda x1, x2, y1, y2: y1,
                lambda x1, x2, y1, y2: y2,
            ]
            if dreal is not None
            else None
        ),
        # Use the same domain-restricted transition semantics as PT/NNT:
        # only transitions with x in X and f(x) in X are quantified.
        # The interval helper above is kept only for old diagnostics.
        next_relation_z3=None,
        verify_backend="dreal",
        max_c2_points=len(pts),
        max_c3_points=len(pts),
        eps_min=1.0,
        eps_max=1.0,
        strengthened=True,
        tau1=1.0,
        tau2=1.0,
        tau3=0.0,
        dreal_precision=dreal_precision,
        boundary_values=[
            [0.0, PI / 9, 5 * PI / 6, 8 * PI / 9],
            [0.0, PI / 9, 5 * PI / 6, 8 * PI / 9],
        ],
        c1_sample_margin=0.0,
        c2_sample_margin=0.0,
        c2_cex_margin=0.0,
        c1_cex_radii=(0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3),
        paper_transition_semantics=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ex2 closure-certificate synthesis")
    parser.add_argument("--out", type=str, default="res_cc_ex2.json")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--grid-step", type=float, default=0.0)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    parser.add_argument("--z3-timeout-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qi", type=int, default=0)
    parser.add_argument("--qj", type=int, default=0)
    args = parser.parse_args()

    print_header("ex2", "CC", "closure_certificate", {"max_iter": args.max_iter, "solver_synth": "z3-lra", "solver_verify": "dreal", "transition_semantics": "domain_restricted"})
    start = time.time()
    result = synthesize_closure(build_case(dreal_precision=args.dreal_precision), max_iter=args.max_iter)
    result.update({"example": "ex2", "method": "CC", "certificate_type": "closure_certificate", "elapsed_sec": time.time() - start})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))


if __name__ == "__main__":
    main()
