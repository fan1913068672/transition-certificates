from __future__ import annotations

from dataclasses import dataclass
import os
import time
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
    # Optional exact/over-approximating Z3 transition relation.  This is not
    # used by the paper-faithful CC reproduction; it is kept only so old
    # diagnostic variants can still be inspected explicitly.
    #
    # If provided,
    # the Z3 verifier declares fresh next-state variables xn and asserts this
    # relation instead of substituting next_z3(x).  This is useful for sound
    # interval over-approximations of small transcendental terms, e.g. replacing
    # k*sin(.) by an interval uncertainty.  The final obligation becomes
    # stronger: it is checked for every next state admitted by the relation.
    next_relation_z3: Callable[[tuple[object, ...], tuple[object, ...]], object] | None = None
    # Optional exact specialization of template terms by the label of the
    # first state argument of T_{ij}(x,y).  This is an SMT-performance device:
    # if a template contains AP-indicator terms such as If(x in VF, 1, 0),
    # verification can be split by the AP label of x and those indicators can
    # be replaced by constants.  This is logically equivalent when the label
    # predicates form a partition of the domain; it does not weaken any proof
    # obligation.
    template_terms_z3_by_first_label: dict[Label, list[Callable[..., object]]] | None = None
    x0_labels: list[Label] | None = None
    verify_backend: str = "dreal"
    coeff_bound: float = 20.0
    eps_min: float = 1e-3
    eps_max: float = 5.0
    max_c2_points: int = 48
    max_c3_points: int = 24
    dreal_precision: float = 1e-4
    # Candidate-generation strengthening from Murali et al. HSCC'24,
    # Eq. (24)--(25).  This is only used in the finite-sample synthesis
    # query.  The final check in _find_cex_* still verifies the original
    # closure-certificate implications, so formal soundness does not rely
    # on the strengthening.
    strengthened: bool = False
    tau1: float = 1.0
    tau2: float = 1.0
    tau3: float = 0.0
    # Optional AP/domain boundary coordinates used only to enrich CEGIS
    # counterexample samples.  This does not change the certificate theorem:
    # final acceptance still goes through _find_cex_* over the original
    # continuous proof obligations.
    boundary_values: list[list[float]] | None = None
    c1_extra_points: list[Point] | None = None
    c1_sample_margin: float = 0.0
    c2_sample_margin: float = 0.0
    c2_cex_margin: float = 0.0
    c1_cex_radii: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
    # Domain-restricted transition semantics used by the released benchmarks:
    # closure obligations are checked only for transitions whose source and
    # successor both lie in X, i.e. x in X and f(x) in X.  This matches the
    # transition systems used by PT/NNT and avoids implicitly assuming
    # f(X) subset X when a benchmark box is not forward invariant.
    #
    # Setting this to True keeps the literal paper convention x in X and
    # x' in f(x) without an explicit x' in X guard; use it only for diagnostics
    # when f(X) subset X has been established separately.
    paper_transition_semantics: bool = False


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


def _dedup_points(points: list[Point]) -> list[Point]:
    seen: set[tuple[float, ...]] = set()
    out: list[Point] = []
    for p in points:
        key = tuple(round(float(v), 12) for v in p)
        if key not in seen:
            seen.add(key)
            out.append(tuple(float(v) for v in p))
    return out


def _c1_cex_points(case: ClosureCase, x: Point) -> list[Point]:
    """Generate a small C1-only neighborhood around a formal counterexample.

    dReal/Z3 often return AP-boundary counterexamples with tiny numerical
    perturbations.  Adding only that single floating-point point may let CEGIS
    find the same boundary again.  We therefore add a tiny local cloud and snap
    coordinates to known AP/domain boundaries.  This is a synthesis heuristic
    only; soundness is still provided by the final global verifier.
    """
    base = tuple(float(v) for v in x)
    candidates: list[Point] = [base]
    for r in case.c1_cex_radii:
        if r == 0.0:
            continue
        for dim in range(case.dim):
            for sign in (-1.0, 1.0):
                p = list(base)
                p[dim] += sign * r
                candidates.append(tuple(p))

    boundary_values = case.boundary_values or [
        [lo, hi] for lo, hi in case.domain_bounds
    ]
    snap_tol = max(case.c1_cex_radii) * 20.0 if case.c1_cex_radii else 0.0
    snap_tol = max(snap_tol, 1e-8)
    for dim, values in enumerate(boundary_values):
        for b in values:
            if abs(base[dim] - b) <= snap_tol:
                p = list(base)
                p[dim] = float(b)
                candidates.append(tuple(p))
                for r in case.c1_cex_radii:
                    if r == 0.0:
                        continue
                    for sign in (-1.0, 1.0):
                        q = list(p)
                        q[dim] += sign * r
                        candidates.append(tuple(q))

    return _dedup_points([p for p in candidates if case.domain_member(*p)])


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
            if not case.paper_transition_semantics:
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
                if not case.paper_transition_semantics:
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

    def Td(i, j, x, y, first_label: Label | None = None):
        terms = case.template_terms_z3
        if first_label is not None and case.template_terms_z3_by_first_label is not None:
            terms = case.template_terms_z3_by_first_label.get(first_label, terms)
        return sum(coeff_vals[(i, j)][k] * terms[k](*x, *y) for k in range(len(terms)))

    def add_next_relation(s: z3.Solver, x: tuple[object, ...], prefix: str = "xn") -> tuple[object, ...]:
        if case.next_relation_z3 is None:
            return case.next_z3(*x)
        xn = _make_z3_vars(prefix, case.dim)
        s.add(case.next_relation_z3(x, xn))
        return xn

    def eval_point(m: z3.ModelRef, vars_: tuple[object, ...]) -> Point:
        return tuple(_as_float(m.evaluate(v, model_completion=True)) for v in vars_)

    labels = list(case.label_cond_z3.keys())

    for (qi, lab), dsts in case.automaton.transitions.items():
        for qj in dsts:
            s = z3.SolverFor("QF_NRA")
            x = _make_z3_vars("x", case.dim)
            xn = add_next_relation(s, x, "xn")
            s.add(case.domain_cond_z3(*x), case.label_cond_z3[lab](*x))
            if not case.paper_transition_semantics:
                s.add(case.domain_cond_z3(*xn))
            s.add(Td(qi, qj, x, xn, lab) < 0)
            if s.check() == z3.sat:
                m = s.model()
                xv = eval_point(m, x)
                xnv = eval_point(m, xn) if case.next_relation_z3 is not None else case.next_num(*xv)
                return ("c1", qi, qj, xv, xnv)

    for (qi, lab), dsts in case.automaton.transitions.items():
        for qip in dsts:
            for qj in q_states:
                if case.paper_transition_semantics:
                    s = z3.SolverFor("QF_NRA")
                    x = _make_z3_vars("x", case.dim)
                    y = _make_z3_vars("y", case.dim)
                    xn = add_next_relation(s, x, "xn")
                    s.add(case.domain_cond_z3(*x), case.domain_cond_z3(*y))
                    s.add(case.label_cond_z3[lab](*x))
                    s.add(Td(qip, qj, xn, y, None) >= 0, Td(qi, qj, x, y, lab) < 0)
                    if s.check() == z3.sat:
                        m = s.model()
                        xv = eval_point(m, x)
                        xnv = eval_point(m, xn) if case.next_relation_z3 is not None else case.next_num(*xv)
                        yv = eval_point(m, y)
                        return ("c2", qi, qip, qj, xv, xnv, yv)
                else:
                    for lab_next in labels:
                        s = z3.SolverFor("QF_NRA")
                        x = _make_z3_vars("x", case.dim)
                        y = _make_z3_vars("y", case.dim)
                        xn = add_next_relation(s, x, "xn")
                        s.add(case.domain_cond_z3(*x), case.domain_cond_z3(*y), case.domain_cond_z3(*xn))
                        s.add(case.label_cond_z3[lab](*x), case.label_cond_z3[lab_next](*xn))
                        s.add(Td(qip, qj, xn, y, lab_next) >= 0, Td(qi, qj, x, y, lab) < 0)
                        if s.check() == z3.sat:
                            m = s.model()
                            xv = eval_point(m, x)
                            xnv = eval_point(m, xn) if case.next_relation_z3 is not None else case.next_num(*xv)
                            yv = eval_point(m, y)
                            return ("c2", qi, qip, qj, xv, xnv, yv)

    for ell in case.accepting_states:
        for ellp in case.accepting_states:
            for lab_x0 in (case.x0_labels or labels):
                for lab_y in labels:
                    s = z3.SolverFor("QF_NRA")
                    x0 = _make_z3_vars("x0", case.dim)
                    y = _make_z3_vars("y", case.dim)
                    yp = _make_z3_vars("yp", case.dim)
                    s.add(case.x0_cond_z3(*x0), case.label_cond_z3[lab_x0](*x0))
                    s.add(case.domain_cond_z3(*y), case.label_cond_z3[lab_y](*y), case.domain_cond_z3(*yp))
                    s.add(Td(case.q0, ell, x0, y, lab_x0) >= 0, Td(ell, ellp, y, yp, lab_y) >= 0)
                    s.add(Td(case.q0, ellp, x0, yp, lab_x0) > Td(case.q0, ell, x0, y, lab_x0) - eps_val)
                    if s.check() == z3.sat:
                        m = s.model()
                        x0v = eval_point(m, x0)
                        yv = eval_point(m, y)
                        ypv = eval_point(m, yp)
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
    # The synthesis query is over certificate coefficients only.  All program
    # states in these constraints are concrete CEGIS samples, so every template
    # term is a numeric constant and the constraints are linear in the unknown
    # coefficients/epsilon.  Use LRA here; the nonlinear continuous obligations
    # are still checked by _find_cex_z3/_find_cex_dreal.
    solver = z3.SolverFor("QF_LRA")

    for pair_coeffs in coeffs.values():
        for c in pair_coeffs:
            solver.add(c >= -case.coeff_bound, c <= case.coeff_bound)
    solver.add(eps >= case.eps_min, eps <= case.eps_max)

    pts = list(case.sample_points)
    c1_pts = _dedup_points(pts + list(case.c1_extra_points or []))
    c2_pts = pts[: case.max_c2_points]
    c3_pts = pts[: case.max_c3_points]

    def Tz(i: int, j: int, x: Point, y: Point):
        return z3.simplify(sum(coeffs[(i, j)][k] * case.template_terms_num[k](*x, *y) for k in range(len(case.template_terms_num))))

    def delta_num(i: int, x: Point) -> list[int]:
        lab = case.label_of(*x)
        return case.automaton.transitions.get((i, lab), [])

    for x in c1_pts:
        xn = case.next_num(*x)
        if (not case.paper_transition_semantics) and (not case.domain_member(*xn)):
            continue
        for i in q_states:
            for ip in delta_num(i, x):
                solver.add(Tz(i, ip, x, xn) >= case.c1_sample_margin)

    for x in c2_pts:
        xn = case.next_num(*x)
        if (not case.paper_transition_semantics) and (not case.domain_member(*xn)):
            continue
        for i in q_states:
            for ip in delta_num(i, x):
                for y in c2_pts:
                    for j in q_states:
                        if case.strengthened:
                            solver.add(Tz(i, j, x, y) >= case.tau1 * Tz(ip, j, xn, y) + case.c2_sample_margin)
                        else:
                            solver.add(z3.Implies(Tz(ip, j, xn, y) >= 0, Tz(i, j, x, y) >= 0))

    for x0 in case.x0_samples:
        for ell in case.accepting_states:
            for ellp in case.accepting_states:
                for y in c3_pts:
                    for yp in c3_pts:
                        if case.strengthened:
                            lhs = Tz(case.q0, ell, x0, y) - eps - Tz(case.q0, ellp, x0, yp)
                            rhs = case.tau2 * Tz(case.q0, ell, x0, y) + case.tau3 * Tz(ell, ellp, y, yp)
                            solver.add(lhs >= rhs)
                        else:
                            premise = z3.And(Tz(case.q0, ell, x0, y) >= 0, Tz(ell, ellp, y, yp) >= 0)
                            concl = Tz(case.q0, ellp, x0, yp) <= Tz(case.q0, ell, x0, y) - eps
                            solver.add(z3.Implies(premise, concl))

    iteration = 0
    last_coeff_vals = None
    last_eps_val = None
    last_ce = None
    verbose = os.environ.get("CLOSURE_VERBOSE", "").lower() in {"1", "true", "yes"}
    while iteration < max_iter:
        t_check = time.time()
        check_result = solver.check()
        if verbose:
            print(f"[CC-CEGIS] {case.name}: iteration {iteration + 1}/{max_iter} synth_check={check_result} time={time.time() - t_check:.4f}s", flush=True)
        if check_result != z3.sat:
            break

        model = solver.model()
        coeff_vals = {
            (i, j): [_as_float(model.evaluate(c, model_completion=True)) for c in coeffs[(i, j)]]
            for i in q_states
            for j in q_states
        }
        coeff_vals_z3 = {
            (i, j): [model.evaluate(c, model_completion=True) for c in coeffs[(i, j)]]
            for i in q_states
            for j in q_states
        }
        eps_val = _as_float(model.evaluate(eps, model_completion=True))
        eps_val_z3 = model.evaluate(eps, model_completion=True)
        last_coeff_vals = coeff_vals
        last_eps_val = eps_val

        t_verify = time.time()
        if case.verify_backend == "dreal":
            ce = _find_cex_dreal(case, coeff_vals, eps_val)
            verify_backend = "dreal"
        elif case.verify_backend == "z3":
            ce = _find_cex_z3(case, coeff_vals_z3, eps_val_z3)
            verify_backend = "z3"
        else:
            raise ValueError(f"unknown verify backend: {case.verify_backend}")
        if verbose:
            ce_kind = "none" if ce is None else str(ce[0])
            print(f"[CC-CEGIS] {case.name}: verify={verify_backend} ce={ce_kind} time={time.time() - t_verify:.4f}s", flush=True)
            if ce is not None:
                print(f"[CC-CEGIS] {case.name}: ce_detail={ce}", flush=True)

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

        last_ce = ce

        if ce[0] == "c1":
            _, i, ip, x, xn = ce
            if case.domain_member(*xn):
                solver.add(Tz(i, ip, x, xn) >= case.c1_sample_margin)
            if case.next_relation_z3 is not None:
                # For relation-based verification the counterexample contains
                # an arbitrary admitted successor, not necessarily next_num(x).
                # Add a small x/y cloud around the returned pair.  This is only
                # a CEGIS strengthening; the final proof is still the global Z3
                # relation check.
                for xp in _c1_cex_points(case, x):
                    if ip not in delta_num(i, xp):
                        continue
                    for xnp in _c1_cex_points(case, xn):
                        solver.add(Tz(i, ip, xp, xnp) >= case.c1_sample_margin)
            else:
                for xp in _c1_cex_points(case, x):
                    xpn = case.next_num(*xp)
                    if case.domain_member(*xpn):
                        solver.add(Tz(i, ip, xp, xpn) >= case.c1_sample_margin)
        elif ce[0] == "c2":
            _, i, ip, j, x, xn, y = ce
            if case.strengthened:
                solver.add(Tz(i, j, x, y) >= case.tau1 * Tz(ip, j, xn, y) + case.c2_cex_margin)
            else:
                solver.add(z3.Implies(Tz(ip, j, xn, y) >= 0, Tz(i, j, x, y) >= 0))
        else:
            _, ell, ellp, x0, y, yp = ce
            if case.strengthened:
                lhs = Tz(case.q0, ell, x0, y) - eps - Tz(case.q0, ellp, x0, yp)
                rhs = case.tau2 * Tz(case.q0, ell, x0, y) + case.tau3 * Tz(ell, ellp, y, yp)
                solver.add(lhs >= rhs)
            else:
                premise = z3.And(Tz(case.q0, ell, x0, y) >= 0, Tz(ell, ellp, y, yp) >= 0)
                concl = Tz(case.q0, ellp, x0, yp) <= Tz(case.q0, ell, x0, y) - eps
                solver.add(z3.Implies(premise, concl))

        iteration += 1

    return {
        "success": False,
        "iterations": iteration,
        "epsilon": None,
        "coefficients": None,
        "last_epsilon": last_eps_val,
        "last_counterexample": repr(last_ce),
        "last_coefficients": (
            {f"T_{i}_{j}": last_coeff_vals[(i, j)] for i in q_states for j in q_states}
            if last_coeff_vals is not None
            else None
        ),
        "backend": {"synth": "z3", "verify": case.verify_backend},
    }
