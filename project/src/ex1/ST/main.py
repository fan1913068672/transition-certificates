from __future__ import annotations

import argparse
import json
import math
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

TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926


def in_x(x: float) -> bool:
    return 0.0 <= x <= 2 * PI


def in_x_cond(x):
    return dreal.And(x >= 0, x <= 2 * PI) if dreal is not None else z3.And(x >= 0, x <= 2 * PI)


def in_x0(x: float) -> bool:
    return 4 * PI / 9 <= x <= 5 * PI / 9


def in_x0_cond(x):
    return dreal.And(x >= 4 * PI / 9, x <= 5 * PI / 9) if dreal is not None else z3.And(x >= 4 * PI / 9, x <= 5 * PI / 9)


def in_unsafe(x: float) -> bool:
    return 7 * PI / 9 <= x <= 8 * PI / 9


def in_unsafe_cond(x):
    return dreal.And(x >= 7 * PI / 9, x <= 8 * PI / 9) if dreal is not None else z3.And(x >= 7 * PI / 9, x <= 8 * PI / 9)


def delta(x: float, q: int) -> list[int]:
    if q == 1:
        return [0] if in_unsafe(x) else [1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid q: {q}")


def f_m(x: float) -> float:
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x**2 + 1.69


def f_t(x):
    if dreal is None:
        return x + TS * OMEGA + 1.69 - 0.532 * x**2
    return x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x**2 + 1.69


def step_sample(a: float, b: float, s: float) -> list[float]:
    vals: list[float] = []
    n0 = int(round(a / s))
    n1 = int(round(b / s))
    for i in range(n0, n1 + 1):
        vals.append(round(i * s, 10))
    return vals


def build_case() -> CaseModel:
    automaton = Automaton(
        states=[0, 1, 2, 3],
        initial_states=[0],
        accepting_states=[2],
        transitions={
            (0, label()): [3],
            (0, label("i")): [1],
            (0, label("u")): [3],
            (0, label("i", "u")): [2],
            (1, label()): [1],
            (1, label("i")): [1],
            (1, label("u")): [2],
            (1, label("i", "u")): [2],
            (2, label()): [2],
            (2, label("i")): [2],
            (2, label("u")): [2],
            (2, label("i", "u")): [2],
            (3, label()): [3],
            (3, label("i")): [3],
            (3, label("u")): [3],
            (3, label("i", "u")): [3],
        },
    )

    return CaseModel(
        name="ex1",
        automaton=automaton,
        dim=1,
        domain_cond=lambda x: in_x_cond(x),
        next_cond=lambda x: (f_t(x),),
        next_num=lambda x: (f_m(x),),
        label_cond={
            label(): lambda x: dreal.And(in_x_cond(x), dreal.Not(in_x0_cond(x)), dreal.Not(in_unsafe_cond(x))) if dreal is not None else z3.And(in_x_cond(x), z3.Not(in_x0_cond(x)), z3.Not(in_unsafe_cond(x))),
            label("i"): lambda x: dreal.And(in_x0_cond(x), dreal.Not(in_unsafe_cond(x))) if dreal is not None else z3.And(in_x0_cond(x), z3.Not(in_unsafe_cond(x))),
            label("u"): lambda x: dreal.And(in_unsafe_cond(x), dreal.Not(in_x0_cond(x))) if dreal is not None else z3.And(in_unsafe_cond(x), z3.Not(in_x0_cond(x))),
            label("i", "u"): lambda x: dreal.And(in_x0_cond(x), in_unsafe_cond(x)) if dreal is not None else z3.And(in_x0_cond(x), in_unsafe_cond(x)),
        },
        label_member={
            label(): lambda x: in_x(x) and (not in_x0(x)) and (not in_unsafe(x)),
            label("i"): lambda x: in_x0(x) and (not in_unsafe(x)),
            label("u"): lambda x: in_unsafe(x) and (not in_x0(x)),
            label("i", "u"): lambda x: in_x0(x) and in_unsafe(x),
        },
        sample_points=[(x,) for x in step_sample(0.0, 2 * PI, 0.05)],
        template_terms_z3=[
            lambda x: 1.0,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: x**4,
        ],
        template_terms_dreal=[
            lambda x: 1.0,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: x**4,
        ],
        template_terms_num=[
            lambda x: 1.0,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: x**4,
        ],
        verify_with_dreal=True,
        domain_cond_z3=lambda x: z3.And(x >= 0, x <= 2 * PI),
        label_cond_z3={
            label(): lambda x: z3.And(
                x >= 0,
                x <= 2 * PI,
                z3.Not(z3.And(x >= 4 * PI / 9, x <= 5 * PI / 9)),
                z3.Not(z3.And(x >= 7 * PI / 9, x <= 8 * PI / 9)),
            ),
            label("i"): lambda x: z3.And(
                x >= 4 * PI / 9,
                x <= 5 * PI / 9,
                z3.Not(z3.And(x >= 7 * PI / 9, x <= 8 * PI / 9)),
            ),
            label("u"): lambda x: z3.And(
                x >= 7 * PI / 9,
                x <= 8 * PI / 9,
                z3.Not(z3.And(x >= 4 * PI / 9, x <= 5 * PI / 9)),
            ),
            label("i", "u"): lambda x: z3.And(
                x >= 4 * PI / 9,
                x <= 5 * PI / 9,
                x >= 7 * PI / 9,
                x <= 8 * PI / 9,
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ex1 strict Wong state-triplet baseline")
    parser.add_argument("--out", type=str, default="res_st_ex1.json")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    args = parser.parse_args()

    print_header("ex1", "ST", "state_triplet_barrier", {"strict_wong": True, "backend": "z3+dreal", "max_iter": args.max_iter, "dreal_precision": args.dreal_precision})
    start = time.time()
    case = build_case()
    case.dreal_precision = args.dreal_precision
    result = analyze_case(case, max_iter=args.max_iter)
    result.update({"example": "ex1", "method": "ST", "certificate_type": "state_triplet_barrier"})
    elapsed = time.time() - start
    result["elapsed_sec"] = elapsed

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result["success"]), result.get("iterations"), elapsed, str(out_path))


if __name__ == "__main__":
    main()
