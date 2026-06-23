from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import z3
import dreal
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

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
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x**2 + 1.69


def f_dreal(x):
    return x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x**2 + 1.69


def step_sample(a: float, b: float, s: float):
    vals = []
    n0 = int(round(a / s))
    n1 = int(round(b / s))
    for i in range(n0, n1 + 1):
        vals.append(round(i * s, 10))
    return vals


def build_case(dreal_precision: float = 1e-4):
    """Compatibility hook for the existing structural test.

    The actual ex1 CC implementation below follows the original paper's
    safety closure certificate, not the product-NBA LTL certificate.  The
    automaton object here is therefore metadata-only for consistency checks.
    """

    return SimpleNamespace(
        accepting_states=[0],
        verify_backend="dreal",
        automaton=SimpleNamespace(
            states=[0, 1],
            initial_states=[1],
            accepting_states=[0],
            transitions={
                (1, ()): [1],
                (1, ("a",)): [0],
                (0, ()): [0],
                (0, ("a",)): [0],
            }
        ),
    )


def _as_float(v: z3.ExprRef) -> float:
    s = v.as_decimal(30)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def _t_num(c, x, y):
    return c[0] + c[1] * x + c[2] * y


def _t_z3(c, x, y):
    return c[0] + c[1] * x + c[2] * y


def _t_dreal(c, x, y):
    return c[0] + c[1] * x + c[2] * y


def _find_cex_dreal(coeffs, eps_val: float, dreal_precision: float):
    """Check original safety closure-certificate conditions.

    This intentionally verifies the original implications, not just the
    strengthened LP constraints used for candidate generation.
    """

    # C1: T(x, f(x)) >= 0 for all domain-restricted transitions
    #     x in X and f(x) in X.
    x = dreal.Variable("x")
    ctx = dreal.Context()
    ctx.SetLogic(dreal.Logic.QF_NRA)
    ctx.config.precision = dreal_precision
    ctx.DeclareVariable(x, 0.0, 2 * PI)
    ctx.Assert(in_x_dreal(x))
    ctx.Assert(in_x_dreal(f_dreal(x)))
    ctx.Assert(_t_dreal(coeffs, x, f_dreal(x)) < 0)
    m = ctx.CheckSat()
    if m is not None:
        xv = float(m[x].mid())
        return ("c1", xv, f_num(xv))

    # C2: T(f(x), y) >= 0 => T(x, y) >= 0 for
    #     x,y in X and f(x) in X.
    x = dreal.Variable("x")
    y = dreal.Variable("y")
    ctx = dreal.Context()
    ctx.SetLogic(dreal.Logic.QF_NRA)
    ctx.config.precision = dreal_precision
    ctx.DeclareVariable(x, 0.0, 2 * PI)
    ctx.DeclareVariable(y, 0.0, 2 * PI)
    ctx.Assert(in_x_dreal(x))
    ctx.Assert(in_x_dreal(y))
    ctx.Assert(in_x_dreal(f_dreal(x)))
    ctx.Assert(_t_dreal(coeffs, f_dreal(x), y) >= 0)
    ctx.Assert(_t_dreal(coeffs, x, y) < 0)
    m = ctx.CheckSat()
    if m is not None:
        xv = float(m[x].mid())
        yv = float(m[y].mid())
        return ("c2", xv, f_num(xv), yv)

    # C3: T(x0, xu) <= -eps for all x0 in X0 and xu in Xu.
    x0 = dreal.Variable("x0")
    xu = dreal.Variable("xu")
    ctx = dreal.Context()
    ctx.SetLogic(dreal.Logic.QF_NRA)
    ctx.config.precision = dreal_precision
    ctx.DeclareVariable(x0, 4 * PI / 9, 5 * PI / 9)
    ctx.DeclareVariable(xu, 7 * PI / 9, 8 * PI / 9)
    ctx.Assert(in_x0_dreal(x0))
    ctx.Assert(in_unsafe_dreal(xu))
    ctx.Assert(_t_dreal(coeffs, x0, xu) > -eps_val)
    m = ctx.CheckSat()
    if m is not None:
        x0v = float(m[x0].mid())
        xuv = float(m[xu].mid())
        return ("c3", x0v, xuv)

    return None


def synthesize_safety_closure(max_iter: int, sample_count: int, dreal_precision: float):
    # Linear safety CC template from Murali et al.: T(x,y)=c0+c1*x+c2*y.
    coeffs = [z3.Real(f"c{i}") for i in range(3)]
    eps = z3.Real("eps")
    solver = z3.SolverFor("QF_LRA")
    for c in coeffs:
        solver.add(c >= -20, c <= 20)
    solver.add(eps >= 1e-3, eps <= 5.0)

    xs = step_sample(0.0, 2 * PI, 2 * PI / max(2, sample_count - 1))
    x0s = sorted(set([x for x in xs if in_x0(x)] + [4 * PI / 9, 5 * PI / 9]))
    xus = sorted(set([x for x in xs if in_unsafe(x)] + [7 * PI / 9, 8 * PI / 9]))

    c1_margin = 0.0
    c3_margin = 0.0

    def add_c1(x):
        if not in_x(f_num(x)):
            return
        solver.add(_t_z3(coeffs, x, f_num(x)) >= c1_margin)

    def add_c2(x, y):
        if not in_x(f_num(x)):
            return
        # Paper Eq. (24) with tau1=1: T(x,y) >= T(f(x),y).
        solver.add(_t_z3(coeffs, x, y) >= _t_z3(coeffs, f_num(x), y))

    def add_c3(x0, xu):
        solver.add(_t_z3(coeffs, x0, xu) <= -eps - c3_margin)

    for x in xs:
        add_c1(x)
    for x in xs:
        for y in xs:
            add_c2(x, y)
    for x0 in x0s:
        for xu in xus:
            add_c3(x0, xu)

    iteration = 0
    while iteration < max_iter and solver.check() == z3.sat:
        m = solver.model()
        cvals = [_as_float(m.evaluate(c, model_completion=True)) for c in coeffs]
        eps_val = _as_float(m.evaluate(eps, model_completion=True))
        ce = _find_cex_dreal(cvals, eps_val, dreal_precision)
        if ce is None:
            return {
                "success": True,
                "iterations": iteration + 1,
                "epsilon": eps_val,
                "coefficients": {"T": cvals, "T_0_0": cvals},
                "automaton": {"metadata_only": True, "accepting_states": [0]},
                "template": "T(x,y)=c0+c1*x+c2*y",
                "certificate_scope": "safety_closure_certificate",
                "backend": {"synth": "z3-lra-paper-cegis", "verify": "dreal"},
                "formal_conditions": [
                    "forall x in X with f(x) in X: T(x,f(x)) >= 0",
                    "forall x,y in X with f(x) in X: T(f(x),y) >= 0 => T(x,y) >= 0",
                    "forall x0 in X0,xu in Xu: T(x0,xu) <= -epsilon",
                ],
            }
        if ce[0] == "c1":
            _, x, _xn = ce
            add_c1(x)
        elif ce[0] == "c2":
            _, x, _xn, y = ce
            add_c2(x, y)
        else:
            _, x0, xu = ce
            add_c3(x0, xu)
        iteration += 1

    return {
        "success": False,
        "iterations": iteration,
        "epsilon": None,
        "coefficients": None,
        "template": "T(x,y)=c0+c1*x+c2*y",
        "certificate_scope": "safety_closure_certificate",
        "backend": {"synth": "z3-lra-paper-cegis", "verify": "dreal"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ex1 safety closure-certificate synthesis")
    parser.add_argument("--out", type=str, default="res_cc_ex1.json")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--grid-step", type=float, default=0.0)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--dreal-precision", type=float, default=1e-4)
    parser.add_argument("--z3-timeout-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qi", type=int, default=0)
    parser.add_argument("--qj", type=int, default=0)
    args = parser.parse_args()

    print_header(
        "ex1",
        "CC",
        "closure_certificate",
        {
            "scope": "safety",
            "max_iter": args.max_iter,
            "sample_count": args.sample_count,
            "solver_synth": "z3-lra-paper-cegis",
            "solver_verify": "dreal",
            "dreal_precision": args.dreal_precision,
        },
    )
    start = time.time()
    result = synthesize_safety_closure(args.max_iter, args.sample_count, args.dreal_precision)
    result.update(
        {
            "example": "ex1",
            "method": "CC",
            "certificate_type": "closure_certificate",
            "elapsed_sec": time.time() - start,
        }
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(result.get("success")), result.get("iterations"), float(result.get("elapsed_sec", 0.0)), str(out_path))


if __name__ == "__main__":
    main()
