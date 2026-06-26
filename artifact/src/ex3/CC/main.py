from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import z3

sys.path.append(str(Path(__file__).resolve().parents[2]))
from closure_smt import ClosureCase, synthesize_closure
from run_output_utils import print_header, print_result
from state_triplet_smt import Automaton, label


def f(x1: float, x2: float):
    alpha, theta, mu, Th, Te = 0.004, 0.01, 0.15, 40.0, 0.0
    def u(x):
        return 0.59 - 0.011 * x
    return (
        (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * Th * u(x1) + theta * Te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te,
    )


def f_z3(x1, x2):
    alpha, theta, mu, Th, Te = 0.004, 0.01, 0.15, 40.0, 0.0
    u1 = 0.59 - 0.011 * x1
    u2 = 0.59 - 0.011 * x2
    return (
        (1 - 2 * alpha - theta - mu * u1) * x1 + alpha * x2 + mu * Th * u1 + theta * Te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u2) * x2 + mu * Th * u2 + theta * Te,
    )


def in_x(x1: float, x2: float) -> bool:
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x0(x1: float, x2: float) -> bool:
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def in_vf(x1: float, x2: float) -> bool:
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def in_x_z3(x1, x2):
    return z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)


def in_x0_z3(x1, x2):
    return z3.And(x1 >= 21, x1 <= 24, x2 >= 21, x2 <= 24)


def in_vf_z3(x1, x2):
    return z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)


def label_of(x1: float, x2: float):
    a = in_x0(x1, x2)
    b = in_vf(x1, x2)
    if a and b:
        return label("a", "b")
    if a:
        return label("a")
    if b:
        return label("b")
    return label()


def template_terms_z3_for_first_label(first_label):
    b_ind = 1.0 if "b" in first_label else 0.0
    return [
        lambda x1, x2, y1, y2: 1.0,
        lambda x1, x2, y1, y2, b=b_ind: b,
        lambda x1, x2, y1, y2, b=b_ind: y1 * b,
        lambda x1, x2, y1, y2, b=b_ind: y2 * b,
        lambda x1, x2, y1, y2: x1,
        lambda x1, x2, y1, y2: x2,
        lambda x1, x2, y1, y2: y1,
        lambda x1, x2, y1, y2: y2,
        lambda x1, x2, y1, y2: x1 * x1,
        lambda x1, x2, y1, y2: x2 * x2,
        lambda x1, x2, y1, y2: y1 * y1,
        lambda x1, x2, y1, y2: y2 * y2,
        lambda x1, x2, y1, y2: x1 * x2,
        lambda x1, x2, y1, y2: y1 * y2,
        lambda x1, x2, y1, y2: x1 * y1,
        lambda x1, x2, y1, y2: x1 * y2,
        lambda x1, x2, y1, y2: x2 * y1,
        lambda x1, x2, y1, y2: x2 * y2,
    ]


def c1_boundary_points():
    dense = [20.0 + 0.5 * i for i in range(29)]
    vals = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
    radii = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]
    pts = set()
    for x1 in dense:
        for x2 in dense:
            pts.add((x1, x2))
    for v in vals:
        for r in radii:
            candidates = [
                (v, 20.0 + r),
                (20.0 + r, v),
                (v, 26.0 - r),
                (26.0 - r, v),
            ]
            for p in candidates:
                if in_x(*p):
                    pts.add(p)
    # Points just outside VF.  These prevent CEGIS from repeatedly returning
    # dyadic counterexamples on the open side of the b / not-b boundary.
    outside_radii = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    low_vals = [20.0 + r for r in radii]
    high_vals = [26.0 + r for r in outside_radii]
    for h in high_vals:
        for v in vals + low_vals:
            candidates = [
                (h, v),
                (v, h),
            ]
            for p in candidates:
                if in_x(*p):
                    pts.add(p)
    return sorted(pts)


def build_case() -> ClosureCase:
    all_labels = [label(), label("a"), label("b"), label("a", "b")]
    automaton = Automaton(
        states=[0, 1, 2, 3],
        initial_states=[0],
        accepting_states=[1],
        transitions={
            (0, label("a")): [2],
            (0, label("a", "b")): [1],
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
    xs = [20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0]
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0s = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    return ClosureCase(
        name="ex3",
        automaton=automaton,
        dim=2,
        q0=0,
        accepting_states=[1],
        sample_points=pts,
        x0_samples=x0s,
        domain_bounds=[(20.0, 34.0), (20.0, 34.0)],
        domain_member=lambda x1, x2: in_x(x1, x2),
        x0_member=lambda x1, x2: in_x0(x1, x2),
        next_num=lambda x1, x2: f(x1, x2),
        next_z3=lambda x1, x2: f_z3(x1, x2),
        next_dreal=None,
        label_of=lambda x1, x2: label_of(x1, x2),
        label_cond_z3={
            label(): lambda x1, x2: z3.And(in_x_z3(x1, x2), z3.Not(in_x0_z3(x1, x2)), z3.Not(in_vf_z3(x1, x2))),
            label("a"): lambda x1, x2: z3.And(in_x0_z3(x1, x2), z3.Not(in_vf_z3(x1, x2))),
            label("b"): lambda x1, x2: z3.And(in_vf_z3(x1, x2), z3.Not(in_x0_z3(x1, x2))),
            label("a", "b"): lambda x1, x2: z3.And(in_x0_z3(x1, x2), in_vf_z3(x1, x2)),
        },
        label_cond_dreal=None,
        domain_cond_z3=lambda x1, x2: in_x_z3(x1, x2),
        domain_cond_dreal=None,
        x0_cond_z3=lambda x1, x2: in_x0_z3(x1, x2),
        x0_cond_dreal=None,
        template_terms_num=[
            lambda x1, x2, y1, y2: 1.0,
            lambda x1, x2, y1, y2: 1.0 if in_vf(x1, x2) else 0.0,
            lambda x1, x2, y1, y2: y1 * (1.0 if in_vf(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: y2 * (1.0 if in_vf(x1, x2) else 0.0),
            lambda x1, x2, y1, y2: x1,
            lambda x1, x2, y1, y2: x2,
            lambda x1, x2, y1, y2: y1,
            lambda x1, x2, y1, y2: y2,
            lambda x1, x2, y1, y2: x1 * x1,
            lambda x1, x2, y1, y2: x2 * x2,
            lambda x1, x2, y1, y2: y1 * y1,
            lambda x1, x2, y1, y2: y2 * y2,
            lambda x1, x2, y1, y2: x1 * x2,
            lambda x1, x2, y1, y2: y1 * y2,
            lambda x1, x2, y1, y2: x1 * y1,
            lambda x1, x2, y1, y2: x1 * y2,
            lambda x1, x2, y1, y2: x2 * y1,
            lambda x1, x2, y1, y2: x2 * y2,
        ],
        template_terms_z3=[
            lambda x1, x2, y1, y2: 1.0,
            lambda x1, x2, y1, y2: z3.If(in_vf_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y1 * z3.If(in_vf_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: y2 * z3.If(in_vf_z3(x1, x2), 1.0, 0.0),
            lambda x1, x2, y1, y2: x1,
            lambda x1, x2, y1, y2: x2,
            lambda x1, x2, y1, y2: y1,
            lambda x1, x2, y1, y2: y2,
            lambda x1, x2, y1, y2: x1 * x1,
            lambda x1, x2, y1, y2: x2 * x2,
            lambda x1, x2, y1, y2: y1 * y1,
            lambda x1, x2, y1, y2: y2 * y2,
            lambda x1, x2, y1, y2: x1 * x2,
            lambda x1, x2, y1, y2: y1 * y2,
            lambda x1, x2, y1, y2: x1 * y1,
            lambda x1, x2, y1, y2: x1 * y2,
            lambda x1, x2, y1, y2: x2 * y1,
            lambda x1, x2, y1, y2: x2 * y2,
        ],
        template_terms_z3_by_first_label={
            lab: template_terms_z3_for_first_label(lab)
            for lab in all_labels
        },
        x0_labels=[label("a", "b")],
        template_terms_dreal=None,
        verify_backend="z3",
        max_c2_points=16,
        max_c3_points=0,
        eps_min=0.5,
        eps_max=0.5,
        strengthened=True,
        tau1=1.0,
        tau2=0.4,
        tau3=0.1,
        boundary_values=[
            [20.0, 21.0, 24.0, 26.0, 34.0],
            [20.0, 21.0, 24.0, 26.0, 34.0],
        ],
        c1_extra_points=c1_boundary_points(),
        c1_sample_margin=1e-5,
        c2_sample_margin=0.0,
        c2_cex_margin=1e-9,
        c1_cex_radii=(0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3),
        paper_transition_semantics=False,
    )


def delta_state_based_gf(q, x1, x2):
    return build_case().automaton.transitions[(q, label_of(x1, x2))]


def delta_closure_paper(q, x1, x2):
    return delta_state_based_gf(q, x1, x2)


def delta_main_pt(q, x1, x2):
    if q == 0:
        return [1] if in_x0(x1, x2) else [2]
    if q == 1:
        return [1]
    return [2]


def main() -> None:
    parser = argparse.ArgumentParser(description="ex3 closure-certificate synthesis")
    parser.add_argument('--mode', choices=['state', 'pt', 'main', 'closure'], default='state')
    parser.add_argument('--out', default='res_cc_ex3.json')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--grid-step', type=float, default=0.0)
    parser.add_argument('--dreal-precision', type=float, default=0.0)
    parser.add_argument('--z3-timeout-ms', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--qi', type=int, default=0)
    parser.add_argument('--qj', type=int, default=0)
    args = parser.parse_args()

    resolved_mode = 'state' if args.mode in {'state', 'closure'} else 'pt'
    print_header("ex3", "CC", "closure_certificate", {"mode": resolved_mode, "solver_synth": "z3", "solver_verify": "z3", "transition_semantics": "domain_restricted"})
    start = time.time()
    result = synthesize_closure(build_case(), max_iter=args.max_iter)
    result.update({"example": "ex3", "method": "CC", "certificate_type": "closure_certificate", "mode": resolved_mode, "elapsed_sec": time.time() - start})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))


if __name__ == '__main__':
    main()
