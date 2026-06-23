from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import z3
import dreal

sys.path.append(str(Path(__file__).resolve().parents[2]))
from closure_smt import ClosureCase, synthesize_closure
from run_output_utils import print_header, print_result
from state_triplet_smt import Automaton, label

TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926


def in_x(x: float) -> bool:
    return 0.0 <= x <= 2 * PI


def in_x0(x: float) -> bool:
    return 4 * PI / 9 <= x <= 5 * PI / 9


def in_unsafe(x: float) -> bool:
    return 7 * PI / 9 <= x <= 8 * PI / 9


def in_x0_z3(x):
    return z3.And(x >= 4 * PI / 9, x <= 5 * PI / 9)


def in_unsafe_z3(x):
    return z3.And(x >= 7 * PI / 9, x <= 8 * PI / 9)


def in_x_z3(x):
    return z3.And(x >= 0, x <= 2 * PI)


def in_x0_dreal(x):
    return dreal.And(x >= 4 * PI / 9, x <= 5 * PI / 9)


def in_unsafe_dreal(x):
    return dreal.And(x >= 7 * PI / 9, x <= 8 * PI / 9)


def in_x_dreal(x):
    return dreal.And(x >= 0, x <= 2 * PI)


def f_num(x: float):
    return (x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x**2 + 1.69,)


def f_z3(x):
    return (x + TS * OMEGA + 1.69 - 0.532 * x**2,)


def f_dreal(x):
    return (x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x**2 + 1.69,)


def step_sample(a: float, b: float, s: float):
    vals = []
    n0 = int(round(a / s))
    n1 = int(round(b / s))
    for i in range(n0, n1 + 1):
        vals.append(round(i * s, 10))
    return vals


def label_of(x: float):
    if in_unsafe(x):
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
    xs = step_sample(0.0, 2 * PI, 0.1)
    x0s = [x for x in xs if in_x0(x)]
    return ClosureCase(
        name="ex1",
        automaton=automaton,
        dim=1,
        q0=1,
        accepting_states=[0],
        sample_points=[(x,) for x in xs],
        x0_samples=[(x,) for x in x0s],
        domain_bounds=[(0.0, 2 * PI)],
        domain_member=lambda x: in_x(x),
        x0_member=lambda x: in_x0(x),
        next_num=lambda x: f_num(x),
        next_z3=lambda x: f_z3(x),
        next_dreal=lambda x: f_dreal(x),
        label_of=lambda x: label_of(x),
        label_cond_z3={
            label(): lambda x: z3.And(in_x_z3(x), z3.Not(in_unsafe_z3(x))),
            label("a"): lambda x: in_unsafe_z3(x),
        },
        label_cond_dreal={
            label(): lambda x: dreal.And(in_x_dreal(x), dreal.Not(in_unsafe_dreal(x))),
            label("a"): lambda x: in_unsafe_dreal(x),
        },
        domain_cond_z3=lambda x: in_x_z3(x),
        domain_cond_dreal=lambda x: in_x_dreal(x),
        x0_cond_z3=lambda x: in_x0_z3(x),
        x0_cond_dreal=lambda x: in_x0_dreal(x),
        template_terms_num=[
            lambda x, y: 1.0,
            lambda x, y: x,
            lambda x, y: y,
        ],
        template_terms_z3=[
            lambda x, y: 1.0,
            lambda x, y: x,
            lambda x, y: y,
        ],
        template_terms_dreal=[
            lambda x, y: 1.0,
            lambda x, y: x,
            lambda x, y: y,
        ],
        verify_backend="dreal",
        dreal_precision=dreal_precision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ex1 closure-certificate synthesis")
    parser.add_argument("--out", type=str, default="res_cc_ex1.json")
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

    print_header("ex1", "CC", "closure_certificate", {"max_iter": args.max_iter, "solver_synth": "z3", "solver_verify": "dreal", "dreal_precision": args.dreal_precision})
    start = time.time()
    result = synthesize_closure(build_case(dreal_precision=args.dreal_precision), max_iter=args.max_iter)
    result.update({"example": "ex1", "method": "CC", "certificate_type": "closure_certificate", "elapsed_sec": time.time() - start})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))


if __name__ == "__main__":
    main()
