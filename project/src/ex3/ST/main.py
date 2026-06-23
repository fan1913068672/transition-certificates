from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

try:
    import dreal
except ModuleNotFoundError:  # pragma: no cover
    dreal = None
import z3

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result
from state_triplet_smt import Automaton, CaseModel, analyze_case, label


def in_x(x1: float, x2: float) -> bool:
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x_cond(x1, x2):
    if dreal is not None:
        return dreal.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)
    return z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)


def in_x0(x1: float, x2: float) -> bool:
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def in_x0_cond(x1, x2):
    if dreal is not None:
        return dreal.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)
    return z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)


def in_vf(x1: float, x2: float) -> bool:
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def b_cond(x1, x2):
    if dreal is not None:
        return dreal.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)
    return z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)


def label_value(x1: float, x2: float) -> int:
    has_a = in_x0(x1, x2)
    has_b = in_vf(x1, x2)
    if has_a and has_b:
        return 3
    if has_a:
        return 2
    if has_b:
        return 1
    return 0


def f(x1: float, x2: float) -> tuple[float, float]:
    alpha, theta, te, th, mu = 0.004, 0.01, 0.0, 40.0, 0.15

    def u(x: float) -> float:
        return 0.59 - 0.011 * x

    x1n = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te
    x2n = alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te
    return x1n, x2n


def f_cond(x1, x2):
    alpha, theta, te, th, mu = 0.004, 0.01, 0.0, 40.0, 0.15

    def u(x):
        return 0.59 - 0.011 * x

    x1n = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te
    x2n = alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te
    return x1n, x2n


def step_sample(a: int, b: int, s: int) -> list[float]:
    return [float(i) for i in range(a, b + 1, s)]


def build_case() -> CaseModel:
    automaton = Automaton(
        states=[0, 1, 2, 3],
        initial_states=[0],
        accepting_states=[1],
        transitions={
            (0, label("a")): [2],
            (0, label("a", "b")): [2],
            (0, label()): [3],
            (0, label("b")): [3],
            (1, label()): [2],
            (1, label("a")): [2],
            (1, label("b")): [1],
            (1, label("a", "b")): [1],
            (2, label()): [2],
            (2, label("a")): [2],
            (2, label("b")): [1],
            (2, label("a", "b")): [1],
            (3, label()): [3],
            (3, label("a")): [3],
            (3, label("b")): [3],
            (3, label("a", "b")): [3],
        },
    )

    xs = step_sample(20, 34, 1)
    return CaseModel(
        name="ex3",
        automaton=automaton,
        dim=2,
        domain_cond=lambda x1, x2: in_x_cond(x1, x2),
        next_cond=lambda x1, x2: f_cond(x1, x2),
        next_num=lambda x1, x2: f(x1, x2),
        label_cond={
            label(): lambda x1, x2: dreal.And(in_x_cond(x1, x2), dreal.Not(dreal.Or(in_x0_cond(x1, x2), b_cond(x1, x2)))) if dreal is not None else z3.And(in_x_cond(x1, x2), z3.Not(z3.Or(in_x0_cond(x1, x2), b_cond(x1, x2)))),
            label("a"): lambda x1, x2: dreal.And(in_x0_cond(x1, x2), dreal.Not(b_cond(x1, x2))) if dreal is not None else z3.And(in_x0_cond(x1, x2), z3.Not(b_cond(x1, x2))),
            label("b"): lambda x1, x2: dreal.And(b_cond(x1, x2), dreal.Not(in_x0_cond(x1, x2))) if dreal is not None else z3.And(b_cond(x1, x2), z3.Not(in_x0_cond(x1, x2))),
            label("a", "b"): lambda x1, x2: dreal.And(in_x0_cond(x1, x2), b_cond(x1, x2)) if dreal is not None else z3.And(in_x0_cond(x1, x2), b_cond(x1, x2)),
        },
        label_member={
            label(): lambda x1, x2: in_x(x1, x2) and (not in_x0(x1, x2)) and (not in_vf(x1, x2)),
            label("a"): lambda x1, x2: in_x0(x1, x2) and (not in_vf(x1, x2)),
            label("b"): lambda x1, x2: in_vf(x1, x2) and (not in_x0(x1, x2)),
            label("a", "b"): lambda x1, x2: in_x0(x1, x2) and in_vf(x1, x2),
        },
        sample_points=[(x1, x2) for x1 in xs for x2 in xs],
        template_terms_z3=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
            lambda x1, x2: z3.If(x1 >= x2, x1, x2),
        ],
        template_terms_dreal=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
            lambda x1, x2: dreal.if_then_else(x1 >= x2, x1, x2),
        ],
        template_terms_num=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
            lambda x1, x2: max(x1, x2),
        ],
        verify_with_dreal=True,
        formal_backend="z3",
        domain_cond_z3=lambda x1, x2: z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34),
        next_cond_z3=lambda x1, x2: f_cond(x1, x2),
        label_cond_z3={
            label(): lambda x1, x2: z3.And(
                z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34),
                z3.Not(z3.Or(z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24), z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26))),
            ),
            label("a"): lambda x1, x2: z3.And(
                z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24),
                z3.Not(z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)),
            ),
            label("b"): lambda x1, x2: z3.And(
                z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26),
                z3.Not(z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)),
            ),
            label("a", "b"): lambda x1, x2: z3.And(
                z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24),
                z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26),
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ex3 strict Wong state-triplet baseline")
    parser.add_argument("--out", type=str, default="res_st_ex3.json")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    args = parser.parse_args()

    print_header("ex3", "ST", "state_triplet_barrier", {"strict_wong": True, "backend": "z3+z3", "max_iter": args.max_iter, "dreal_precision": args.dreal_precision})
    start = time.time()
    case = build_case()
    case.dreal_precision = args.dreal_precision
    result = analyze_case(case, max_iter=args.max_iter)
    result.update(
        {
            "example": "ex3",
            "method": "ST",
            "certificate_type": "state_triplet_barrier",
            "negated_specification": "a ∧ GF b",
        }
    )
    elapsed = time.time() - start
    result["elapsed_sec"] = elapsed

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result["success"]), result.get("iterations"), elapsed, str(out_path))


if __name__ == "__main__":
    main()
