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

PI = 3.1415926
TS = 0.1
OMEGA = 0.01
K = 0.0006


def in_x(x1: float, x2: float) -> bool:
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x_cond(x1, x2):
    if dreal is not None:
        return dreal.And(dreal.And(x1 >= 0, x1 <= 8 * PI / 9), dreal.And(x2 >= 0, x2 <= 8 * PI / 9))
    return z3.And(z3.And(x1 >= 0, x1 <= 8 * PI / 9), z3.And(x2 >= 0, x2 <= 8 * PI / 9))


def in_x0(x1: float, x2: float) -> bool:
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


def in_x0_cond(x1, x2):
    if dreal is not None:
        return dreal.And(dreal.And(x1 >= 0, x1 <= PI / 9), dreal.And(x2 >= 0, x2 <= PI / 9))
    return z3.And(z3.And(x1 >= 0, x1 <= PI / 9), z3.And(x2 >= 0, x2 <= PI / 9))


def in_unsafe(x1: float, x2: float) -> bool:
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def in_unsafe_cond(x1, x2):
    if dreal is not None:
        return dreal.Or(
            dreal.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
            dreal.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
        )
    return z3.Or(
        z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
        z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
    )


def delta(x1: float, x2: float, q: int) -> list[int]:
    if q == 1:
        return [0] if in_unsafe(x1, x2) else [1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid q: {q}")


def f_m(x1: float, x2: float) -> tuple[float, float]:
    x1p = x1 + TS * OMEGA + 1.69 + TS * K * math.sin(x2 - x1) - 0.532 * x1**2
    x2p = x2 + TS * OMEGA + 1.69 + TS * K * math.sin(x1 - x2) - 0.532 * x2**2
    return x1p, x2p


def f_t(x1, x2):
    if dreal is None:
        x1p = x1 + TS * OMEGA + 1.69 - 0.532 * x1**2
        x2p = x2 + TS * OMEGA + 1.69 - 0.532 * x2**2
        return x1p, x2p
    x1p = x1 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x2 - x1) - 0.532 * x1**2
    x2p = x2 + TS * OMEGA + 1.69 + TS * K * dreal.sin(x1 - x2) - 0.532 * x2**2
    return x1p, x2p


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

    xs = step_sample(0.0, 8 * PI / 9, 0.1)
    return CaseModel(
        name="ex2",
        automaton=automaton,
        dim=2,
        domain_cond=lambda x1, x2: in_x_cond(x1, x2),
        next_cond=lambda x1, x2: f_t(x1, x2),
        next_num=lambda x1, x2: f_m(x1, x2),
        label_cond={
            label(): lambda x1, x2: dreal.And(in_x_cond(x1, x2), dreal.Not(in_x0_cond(x1, x2)), dreal.Not(in_unsafe_cond(x1, x2))) if dreal is not None else z3.And(in_x_cond(x1, x2), z3.Not(in_x0_cond(x1, x2)), z3.Not(in_unsafe_cond(x1, x2))),
            label("i"): lambda x1, x2: dreal.And(in_x0_cond(x1, x2), dreal.Not(in_unsafe_cond(x1, x2))) if dreal is not None else z3.And(in_x0_cond(x1, x2), z3.Not(in_unsafe_cond(x1, x2))),
            label("u"): lambda x1, x2: dreal.And(in_unsafe_cond(x1, x2), dreal.Not(in_x0_cond(x1, x2))) if dreal is not None else z3.And(in_unsafe_cond(x1, x2), z3.Not(in_x0_cond(x1, x2))),
            label("i", "u"): lambda x1, x2: dreal.And(in_x0_cond(x1, x2), in_unsafe_cond(x1, x2)) if dreal is not None else z3.And(in_x0_cond(x1, x2), in_unsafe_cond(x1, x2)),
        },
        label_member={
            label(): lambda x1, x2: in_x(x1, x2) and (not in_x0(x1, x2)) and (not in_unsafe(x1, x2)),
            label("i"): lambda x1, x2: in_x0(x1, x2) and (not in_unsafe(x1, x2)),
            label("u"): lambda x1, x2: in_unsafe(x1, x2) and (not in_x0(x1, x2)),
            label("i", "u"): lambda x1, x2: in_x0(x1, x2) and in_unsafe(x1, x2),
        },
        sample_points=[(x1, x2) for x1 in xs for x2 in xs],
        template_terms_z3=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
        ],
        template_terms_dreal=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
        ],
        template_terms_num=[
            lambda x1, x2: 1.0,
            lambda x1, x2: x1,
            lambda x1, x2: x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1**2,
            lambda x1, x2: x2**2,
        ],
        verify_with_dreal=True,
        domain_cond_z3=lambda x1, x2: z3.And(
            x1 >= 0,
            x1 <= 8 * PI / 9,
            x2 >= 0,
            x2 <= 8 * PI / 9,
        ),
        label_cond_z3={
            label(): lambda x1, x2: z3.And(
                x1 >= 0,
                x1 <= 8 * PI / 9,
                x2 >= 0,
                x2 <= 8 * PI / 9,
                z3.Not(z3.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9)),
                z3.Not(
                    z3.Or(
                        z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
                        z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
                    )
                ),
            ),
            label("i"): lambda x1, x2: z3.And(
                x1 >= 0,
                x1 <= PI / 9,
                x2 >= 0,
                x2 <= PI / 9,
                z3.Not(
                    z3.Or(
                        z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
                        z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
                    )
                ),
            ),
            label("u"): lambda x1, x2: z3.And(
                z3.Or(
                    z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
                    z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
                ),
                z3.Not(z3.And(x1 >= 0, x1 <= PI / 9, x2 >= 0, x2 <= PI / 9)),
            ),
            label("i", "u"): lambda x1, x2: z3.And(
                x1 >= 0,
                x1 <= PI / 9,
                x2 >= 0,
                x2 <= PI / 9,
                z3.Or(
                    z3.And(x1 >= 5 * PI / 6, x1 <= 8 * PI / 9),
                    z3.And(x2 >= 5 * PI / 6, x2 <= 8 * PI / 9),
                ),
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ex2 strict Wong state-triplet baseline")
    parser.add_argument("--out", type=str, default="res_st_ex2.json")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    args = parser.parse_args()

    print_header("ex2", "ST", "state_triplet_barrier", {"strict_wong": True, "backend": "z3+dreal", "max_iter": args.max_iter, "dreal_precision": args.dreal_precision})
    start = time.time()
    case = build_case()
    case.dreal_precision = args.dreal_precision
    result = analyze_case(case, max_iter=args.max_iter)
    result.update({"example": "ex2", "method": "ST", "certificate_type": "state_triplet_barrier"})
    elapsed = time.time() - start
    result["elapsed_sec"] = elapsed

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result["success"]), result.get("iterations"), elapsed, str(out_path))


if __name__ == "__main__":
    main()
