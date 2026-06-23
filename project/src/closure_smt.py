from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import z3

try:
    import dreal
except ModuleNotFoundError:  # pragma: no cover
    dreal = None

from state_triplet_smt import Automaton, Label, Point, label_to_str


@dataclass
class ClosureCase:
    name: str
    automaton: Automaton
    dim: int
    q0: int
    accepting_states: list[int]
    sample_points: list[Point]
    x0_samples: list[Point]
    domain_bounds: list[tuple[float, float]]
    domain_member: Callable[..., bool]
    x0_member: Callable[..., bool]
    next_num: Callable[..., Point]
    next_z3: Callable[..., tuple[object, ...]]
    next_dreal: Callable[..., tuple[object, ...]] | None
    label_of: Callable[..., Label]
    label_cond_z3: dict[Label, Callable[..., object]]
    label_cond_dreal: dict[Label, Callable[..., object]] | None
    domain_cond_z3: Callable[..., object]
    domain_cond_dreal: Callable[..., object] | None
    x0_cond_z3: Callable[..., object]
    x0_cond_dreal: Callable[..., object] | None
    template_terms_num: list[Callable[..., float]]
    template_terms_z3: list[Callable[..., object]]
    template_terms_dreal: list[Callable[..., object]] | None
    verify_backend: str = "dreal"
    coeff_bound: float = 20.0
    eps_min: float = 1e-3
    eps_max: float = 5.0
    max_c2_points: int = 48
    max_c3_points: int = 24
    dreal_precision: float = 1e-4


def _as_float(v: z3.ExprRef) -> float:
    s = v.as_decimal(20)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def _label_guard_z3(case: ClosureCase, qi: int, qj: int, vars_: tuple[object, ...]):
    guards = []
    for (src, lab), dsts in case.automaton.transitions.items():
        if src == qi and qj in dsts:
            guards.append(case.label_cond_z3[lab](*vars_))
    if not guards:
        return z3.BoolVal(False)
    return z3.Or(*guards)


def _label_guard_dreal(case: ClosureCase, qi: int, qj: int, vars_: tuple[object, ...]):
    if dreal is None or case.label_cond_dreal is None:
        raise RuntimeError("dReal backend requested but dreal is unavailable")
    guards = []
    for (src, lab), dsts in case.automaton.transitions.items():
        if src == qi and qj in dsts:
            guards.append(case.label_cond_dreal[lab](*vars_))
    if not guards:
        return dreal.Formula.FALSE()
    if len(guards) == 1:
        return guards[0]
    return dreal.Or(*guards)


def _t_num(case: ClosureCase, coeff_vals: dict[tuple[int, int], list[float]], i: int, j: int, x: Point, y: Point) -> float:
    return sum(coeff_vals[(i, j)][k] * case.template_terms_num[k](*x, *y) for k in range(len(case.template_terms_num)))


def _t_z3(case: ClosureCase, coeffs: dict[tuple[int, int], list[z3.ExprRef]], i: int, j: int, x: Point, y: Point):
    return z3.simplify(sum(coeffs[(i, j)][k] * case.template_terms_z3[k](*x, *y) for k in range(len(case.template_terms_z3))))


def _t_dreal(case: ClosureCase, coeff_vals: dict[tuple[int, int], list[float]], i: int, j: int, x: tuple[object, ...], y: tuple[object, ...]):
    if case.template_terms_dreal is None:
        raise RuntimeError("dReal template terms missing")
    expr = 0.0
    for k in range(len(case.template_terms_dreal)):
        expr = expr + coeff_vals[(i, j)][k] * case.template_terms_dreal[k](*x, *y)
    return expr


def _make_z3_vars(prefix: str, dim: int) -> tuple[object, ...]:
    return tuple(z3.Real(f"{prefix}{i}") for i in range(dim))


def _make_dreal_vars(prefix: str, dim: int) -> tuple[object, ...]:
    if dreal is None:
        raise RuntimeError("dReal is unavailable")
    return tuple(dreal.Variable(f"{prefix}{i}") for i in range(dim))


def _declare_dreal(ctx, bounds: list[tuple[float, float]], vars_: tuple[object, ...]) -> None:
    for var, (lo, hi) in zip(vars_, bounds):
        ctx.DeclareVariable(var, lo, hi)


def _find_cex_dreal(case: ClosureCase, coeff_vals: dict[tuple[int, int], list[float]], eps_val: float):
    if dreal is None or case.next_dreal is None or case.domain_cond_dreal is None or case.x0_cond_dreal is None:
        raise RuntimeError("dReal backend requested but symbols are incomplete")

    q_states = case.automaton.states

    for qi in q_states:
        for qj in q_states:
            x = _make_dreal_vars("x", case.dim)
            ctx = dreal.Context()
            ctx.SetLogic(dreal.Logic.QF_NRA)
            ctx.config.precision = case.dreal_precision
            _declare_dreal(ctx, case.domain_bounds, x)
            xn = case.next_dreal(*x)
            ctx.Assert(case.domain_cond_dreal(*x))
            ctx.Assert(case.domain_cond_dreal(*xn))
            ctx.Assert(_label_guard_dreal(case, qi, qj, x))
            ctx.Assert(_t_dreal(case, coeff_vals, qi, qj, x, xn) < 0)
            m = ctx.CheckSat()
            if m is not None:
                xv = tuple(float(m[v].mid()) for v in x)
                return ("c1", qi, qj, xv, case.next_num(*xv))

    for qi in q_states:
        for qip in q_states:
            for qj in q_states:
                x = _make_dreal_vars("x", case.dim)
                y = _make_dreal_vars("y", case.dim)
                ctx = dreal.Context()
                ctx.SetLogic(dreal.Logic.QF_NRA)
                ctx.config.precision = case.dreal_precision
                _declare_dreal(ctx, case.domain_bounds, x)
                _declare_dreal(ctx, case.domain_bounds, y)
                xn = case.next_dreal(*x)
                ctx.Assert(case.domain_cond_dreal(*x))
                ctx.Assert(case.domain_cond_dreal(*y))
                ctx.Assert(case.domain_cond_dreal(*xn))
                ctx.Assert(_label_guard_dreal(case, qi, qip, x))
                ctx.Assert(_t_dreal(case, coeff_vals, qip, qj, xn, y) >= 0)
                ctx.Assert(_t_dreal(case, coeff_vals, qi, qj, x, y) < 0)
                m = ctx.CheckSat()
                if m is not None:
                    xv = tuple(float(m[v].mid()) for v in x)
                    yv = tuple(float(m[v].mid()) for v in y)
                    return ("c2", qi, qip, qj, xv, case.next_num(*xv), yv)

    for ell in case.accepting_states:
        for ellp in case.accepting_states:
            x0 = _make_dreal_vars("x0", case.dim)
            y = _make_dreal_vars("y", case.dim)
            yp = _make_dreal_vars("yp", case.dim)
            ctx = dreal.Context()
            ctx.SetLogic(dreal.Logic.QF_NRA)
            ctx.config.precision = case.dreal_precision
            _declare_dreal(ctx, case.domain_bounds, x0)
            _declare_dreal(ctx, case.domain_bounds, y)
            _declare_dreal(ctx, case.domain_bounds, yp)
            ctx.Assert(case.x0_cond_dreal(*x0))
            ctx.Assert(case.domain_cond_dreal(*y))
            ctx.Assert(case.domain_cond_dreal(*yp))
            ctx.Assert(_t_dreal(case, coeff_vals, case.q0, ell, x0, y) >= 0)
            ctx.Assert(_t_dreal(case, coeff_vals, ell, ellp, y, yp) >= 0)
            ctx.Assert(_t_dreal(case, coeff_vals, case.q0, ellp, x0, yp) > _t_dreal(case, coeff_vals, case.q0, ell, x0, y) - eps_val)
            m = ctx.CheckSat()
            if m is not None:
                x0v = tuple(float(m[v].mid()) for v in x0)
                yv = tuple(float(m[v].mid()) for v in y)
                ypv = tuple(float(m[v].mid()) for v in yp)
                return ("c3", ell, ellp, x0v, yv, ypv)

    return None


def _find_cex_z3(case: ClosureCase, coeff_vals: dict[tuple[int, int], list[float]], eps_val: float):
    q_states = case.automaton.states

    def Td(i, j, x, y):
        return sum(coeff_vals[(i, j)][k] * case.template_terms_z3[k](*x, *y) for k in range(len(case.template_terms_z3)))

    for qi in q_states:
        for qj in q_states:
            s = z3.SolverFor("QF_NRA")
            x = _make_z3_vars("x", case.dim)
            xn = case.next_z3(*x)
            s.add(case.domain_cond_z3(*x), case.domain_cond_z3(*xn), _label_guard_z3(case, qi, qj, x), Td(qi, qj, x, xn) < 0)
            if s.check() == z3.sat:
                m = s.model()
                xv = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in x)
                return ("c1", qi, qj, xv, case.next_num(*xv))

    for qi in q_states:
        for qip in q_states:
            for qj in q_states:
                s = z3.SolverFor("QF_NRA")
                x = _make_z3_vars("x", case.dim)
                y = _make_z3_vars("y", case.dim)
                xn = case.next_z3(*x)
                s.add(case.domain_cond_z3(*x), case.domain_cond_z3(*y), case.domain_cond_z3(*xn))
                s.add(_label_guard_z3(case, qi, qip, x), Td(qip, qj, xn, y) >= 0, Td(qi, qj, x, y) < 0)
                if s.check() == z3.sat:
                    m = s.model()
                    xv = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in x)
                    yv = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in y)
                    return ("c2", qi, qip, qj, xv, case.next_num(*xv), yv)

    for ell in case.accepting_states:
        for ellp in case.accepting_states:
            s = z3.SolverFor("QF_NRA")
            x0 = _make_z3_vars("x0", case.dim)
            y = _make_z3_vars("y", case.dim)
            yp = _make_z3_vars("yp", case.dim)
            s.add(case.x0_cond_z3(*x0), case.domain_cond_z3(*y), case.domain_cond_z3(*yp))
            s.add(Td(case.q0, ell, x0, y) >= 0, Td(ell, ellp, y, yp) >= 0)
            s.add(Td(case.q0, ellp, x0, yp) > Td(case.q0, ell, x0, y) - eps_val)
            if s.check() == z3.sat:
                m = s.model()
                x0v = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in x0)
                yv = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in y)
                ypv = tuple(_as_float(m.evaluate(v, model_completion=True)) for v in yp)
                return ("c3", ell, ellp, x0v, yv, ypv)

    return None


def synthesize_closure(case: ClosureCase, max_iter: int = 50) -> dict:
    q_states = case.automaton.states
    coeffs = {
        (i, j): [z3.Real(f"c_{i}_{j}_{k}") for k in range(len(case.template_terms_z3))]
        for i in q_states
        for j in q_states
    }
    eps = z3.Real("eps")
    solver = z3.SolverFor("QF_NRA")

    for pair_coeffs in coeffs.values():
        for c in pair_coeffs:
            solver.add(c >= -case.coeff_bound, c <= case.coeff_bound)
    solver.add(eps >= case.eps_min, eps <= case.eps_max)

    pts = list(case.sample_points)
    c2_pts = pts[: case.max_c2_points]
    c3_pts = pts[: case.max_c3_points]

    def Tz(i: int, j: int, x: Point, y: Point):
        return z3.simplify(sum(coeffs[(i, j)][k] * case.template_terms_num[k](*x, *y) for k in range(len(case.template_terms_num))))

    def delta_num(i: int, x: Point) -> list[int]:
        lab = case.label_of(*x)
        return case.automaton.transitions.get((i, lab), [])

    for x in pts:
        xn = case.next_num(*x)
        if not case.domain_member(*xn):
            continue
        for i in q_states:
            for ip in delta_num(i, x):
                solver.add(Tz(i, ip, x, xn) >= 0)

    for x in c2_pts:
        xn = case.next_num(*x)
        if not case.domain_member(*xn):
            continue
        for i in q_states:
            for ip in delta_num(i, x):
                for y in c2_pts:
                    for j in q_states:
                        solver.add(z3.Implies(Tz(ip, j, xn, y) >= 0, Tz(i, j, x, y) >= 0))

    for x0 in case.x0_samples:
        for ell in case.accepting_states:
            for ellp in case.accepting_states:
                for y in c3_pts:
                    for yp in c3_pts:
                        premise = z3.And(Tz(case.q0, ell, x0, y) >= 0, Tz(ell, ellp, y, yp) >= 0)
                        concl = Tz(case.q0, ellp, x0, yp) <= Tz(case.q0, ell, x0, y) - eps
                        solver.add(z3.Implies(premise, concl))

    iteration = 0
    while iteration < max_iter and solver.check() == z3.sat:
        model = solver.model()
        coeff_vals = {
            (i, j): [_as_float(model.evaluate(c, model_completion=True)) for c in coeffs[(i, j)]]
            for i in q_states
            for j in q_states
        }
        eps_val = _as_float(model.evaluate(eps, model_completion=True))

        if case.verify_backend == "dreal":
            ce = _find_cex_dreal(case, coeff_vals, eps_val)
            verify_backend = "dreal"
        elif case.verify_backend == "z3":
            ce = _find_cex_z3(case, coeff_vals, eps_val)
            verify_backend = "z3"
        else:
            raise ValueError(f"unknown verify backend: {case.verify_backend}")

        if ce is None:
            return {
                "success": True,
                "iterations": iteration + 1,
                "epsilon": eps_val,
                "coefficients": {f"T_{i}_{j}": coeff_vals[(i, j)] for i in q_states for j in q_states},
                "automaton": {
                    "states": q_states,
                    "initial_state": case.q0,
                    "accepting_states": case.accepting_states,
                    "transitions": [
                        {"src": src, "label": label_to_str(lab), "dst": dst}
                        for (src, lab), dsts in case.automaton.transitions.items()
                        for dst in dsts
                    ],
                },
                "backend": {"synth": "z3", "verify": verify_backend},
            }

        if ce[0] == "c1":
            _, i, ip, x, xn = ce
            solver.add(Tz(i, ip, x, xn) >= 0)
        elif ce[0] == "c2":
            _, i, ip, j, x, xn, y = ce
            solver.add(z3.Implies(Tz(ip, j, xn, y) >= 0, Tz(i, j, x, y) >= 0))
        else:
            _, ell, ellp, x0, y, yp = ce
            premise = z3.And(Tz(case.q0, ell, x0, y) >= 0, Tz(ell, ellp, y, yp) >= 0)
            concl = Tz(case.q0, ellp, x0, yp) <= Tz(case.q0, ell, x0, y) - eps
            solver.add(z3.Implies(premise, concl))

        iteration += 1

    return {
        "success": False,
        "iterations": iteration,
        "epsilon": None,
        "coefficients": None,
        "backend": {"synth": "z3", "verify": case.verify_backend},
    }
