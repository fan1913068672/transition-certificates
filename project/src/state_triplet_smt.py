from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import z3

try:
    import dreal
except ModuleNotFoundError:  # pragma: no cover
    dreal = None


Label = tuple[str, ...]
Point = tuple[float, ...]


def label(*items: str) -> Label:
    return tuple(sorted(items))


def label_to_str(a: Label) -> str:
    return "{" + ",".join(a) + "}" if a else "{}"


@dataclass
class Automaton:
    states: list[int]
    initial_states: list[int]
    accepting_states: list[int]
    transitions: dict[tuple[int, Label], list[int]]

    def outgoing(self, q: int) -> list[tuple[Label, int]]:
        out: list[tuple[Label, int]] = []
        for (src, lab), dsts in self.transitions.items():
            if src == q:
                for dst in dsts:
                    out.append((lab, dst))
        return out

    def labels_between(self, qi: int, qj: int) -> list[Label]:
        labs: list[Label] = []
        for (src, lab), dsts in self.transitions.items():
            if src == qi and qj in dsts:
                labs.append(lab)
        return labs

    def graph_edges(self) -> list[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for (src, _), dsts in self.transitions.items():
            for dst in dsts:
                edges.add((src, dst))
        return sorted(edges)

    def has_self_loop(self, q: int) -> bool:
        return (q, q) in set(self.graph_edges())


@dataclass
class CaseModel:
    name: str
    automaton: Automaton
    dim: int
    domain_cond: Callable[..., object]
    next_cond: Callable[..., tuple[object, ...]]
    next_num: Callable[..., tuple[float, ...]]
    label_cond: dict[Label, Callable[..., object]]
    label_member: dict[Label, Callable[..., bool]]
    sample_points: list[Point]
    template_terms_z3: list[Callable[..., object]]
    template_terms_dreal: list[Callable[..., object]]
    template_terms_num: list[Callable[..., float]]
    verify_with_dreal: bool = True
    margin: float = 1e-2
    dreal_precision: float = 1e-4
    formal_backend: str = "dreal"
    domain_cond_z3: Callable[..., object] | None = None
    next_cond_z3: Callable[..., tuple[object, ...]] | None = None
    label_cond_z3: dict[Label, Callable[..., object]] | None = None


def enumerate_paths_no_repeated_edges(automaton: Automaton, start: int, goal: int) -> list[list[int]]:
    edges = automaton.graph_edges()
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)

    paths: list[list[int]] = []

    def dfs(node: int, path: list[int], used_edges: set[tuple[int, int]]) -> None:
        if node == goal:
            paths.append(path.copy())
        for nb in adj.get(node, []):
            if nb == node:
                continue
            edge = (node, nb)
            if edge in used_edges:
                continue
            used_edges.add(edge)
            path.append(nb)
            dfs(nb, path, used_edges)
            path.pop()
            used_edges.remove(edge)

    dfs(start, [start], set())
    return paths


def ppath(automaton: Automaton, q_acc: int) -> list[list[int]]:
    res: list[list[int]] = []
    for q0 in automaton.initial_states:
        res.extend(enumerate_paths_no_repeated_edges(automaton, q0, q_acc))
    return res


def pcyc(automaton: Automaton, q_acc: int) -> list[list[int]]:
    res = enumerate_paths_no_repeated_edges(automaton, q_acc, q_acc)
    if not res and automaton.has_self_loop(q_acc):
        return [[q_acc]]
    return res


def pf3(path: list[int]) -> list[tuple[int, int, int]]:
    return [tuple(path[i : i + 3]) for i in range(len(path) - 2)]


def _make_dreal_vars(dim: int) -> tuple[object, ...]:
    if dreal is None:  # pragma: no cover
        raise RuntimeError("dReal is required for state-triplet SMT verification")
    return tuple(dreal.Variable(f"x{i}") for i in range(dim))


def _or_all(xs: list[object]) -> object:
    if len(xs) == 1:
        return xs[0]
    return dreal.Or(*xs)


def _and_all(xs: list[object]) -> object:
    if len(xs) == 1:
        return xs[0]
    return dreal.And(*xs)


def _or_all_z3(xs: list[object]) -> object:
    if len(xs) == 1:
        return xs[0]
    return z3.Or(*xs)


def _and_all_z3(xs: list[object]) -> object:
    if len(xs) == 1:
        return xs[0]
    return z3.And(*xs)


def _region_union_cond(case: CaseModel, labels: list[Label], vars_: tuple[object, ...]) -> object | None:
    if not labels:
        return None
    conds = [case.label_cond[a](*vars_) for a in labels]
    return _or_all(conds)


def _region_union_cond_z3(case: CaseModel, labels: list[Label], vars_: tuple[object, ...]) -> object | None:
    if not labels:
        return None
    label_cond = case.label_cond_z3 or case.label_cond
    conds = [label_cond[a](*vars_) for a in labels]
    return _or_all_z3(conds)


def _region_union_member(case: CaseModel, labels: list[Label], x: Point) -> bool:
    return any(case.label_member[a](*x) for a in labels)


def _is_empty_region(case: CaseModel, labels: list[Label]) -> bool:
    if not labels:
        return True
    if dreal is None:
        # In strict mode, emptiness of a label region is a formal obligation.
        # Sampling can miss a non-empty semialgebraic region, so absence of
        # dReal must not be used to discharge an automaton edge.
        return False
    ctx = dreal.Context()
    ctx.config.precision = case.dreal_precision
    ctx.SetLogic(dreal.Logic.QF_NRA)
    vars_ = _make_dreal_vars(case.dim)
    for i, v in enumerate(vars_):
        lo = min(x[i] for x in case.sample_points)
        hi = max(x[i] for x in case.sample_points)
        ctx.DeclareVariable(v, lo, hi)
    dom = case.domain_cond(*vars_)
    y = _region_union_cond(case, labels, vars_)
    if y is None:
        return True
    ctx.Assert(_and_all([dom, y]))
    return ctx.CheckSat() is None


def _is_empty_intersection(case: CaseModel, labels0: list[Label], labels1: list[Label]) -> bool:
    # State-triplet path blocking needs a cross-step barrier proof, not a same-step
    # label-intersection shortcut. Keep this compatibility hook disabled.
    return False


def synthesize_discrete_barrier(
    case: CaseModel,
    sigma0: list[Label],
    sigma1: list[Label],
    sigma_mid: list[Label] | None = None,
    max_iter: int = 50,
) -> dict:
    sigma_mid = sigma_mid or []
    if not sigma0 or not sigma1:
        return {"success": True, "kind": "vacuous", "reason": "empty source or target label set"}

    y0_samples = [x for x in case.sample_points if _region_union_member(case, sigma0, x)]
    y1_samples = [x for x in case.sample_points if _region_union_member(case, sigma1, x)]
    y_samples = [x for x in case.sample_points if _region_union_member(case, sigma0 + sigma1 + sigma_mid, x)]

    if not y0_samples or not y1_samples:
        return {"success": True, "kind": "vacuous", "reason": "empty source or target region"}

    coeffs = [z3.Real(f"c{i}") for i in range(len(case.template_terms_z3))]
    solver = z3.SolverFor("QF_NRA")

    def b_z3(x: Point):
        return z3.simplify(sum(coeffs[i] * case.template_terms_z3[i](*x) for i in range(len(coeffs))))

    for x in y0_samples:
        solver.add(b_z3(x) <= 0)
    for x in y1_samples:
        solver.add(b_z3(x) >= case.margin)
    for x in y_samples:
        if _region_union_member(case, sigma1, x):
            continue
        xn = case.next_num(*x)
        if _region_union_member(case, sigma0 + sigma1 + sigma_mid, xn):
            solver.add(z3.Implies(b_z3(x) <= 0, b_z3(xn) <= 0))

    def z3_value_to_float(v: object) -> float:
        s = v.as_decimal(30)
        if s.endswith("?"):
            s = s[:-1]
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)

    def b_z3_expr(xs: tuple[object, ...], coeff_vals: list[float]):
        return z3.simplify(sum(coeff_vals[i] * case.template_terms_z3[i](*xs) for i in range(len(coeff_vals))))

    if case.formal_backend == "z3":
        domain_cond_z3 = case.domain_cond_z3 or case.domain_cond
        next_cond_z3 = case.next_cond_z3 or case.next_cond
        vars_z3 = tuple(z3.Real(f"x{i}_ce") for i in range(case.dim))

        def point_from_z3_model(m: z3.ModelRef) -> Point:
            return tuple(z3_value_to_float(m.evaluate(v, model_completion=True)) for v in vars_z3)

        last_coeff_vals: list[float] | None = None
        for it in range(max_iter):
            chk = solver.check()
            if chk != z3.sat:
                return {
                    "success": False,
                    "kind": "smt",
                    "reason": f"z3 synthesis {chk} under accumulated sampled/CE constraints",
                    "iterations": it,
                    "last_coefficients": last_coeff_vals,
                }

            model = solver.model()
            coeff_vals = [z3_value_to_float(model.evaluate(c, model_completion=True)) for c in coeffs]
            last_coeff_vals = coeff_vals

            dom = domain_cond_z3(*vars_z3)
            y0 = _region_union_cond_z3(case, sigma0, vars_z3)
            y1 = _region_union_cond_z3(case, sigma1, vars_z3)
            y = _region_union_cond_z3(case, sigma0 + sigma1 + sigma_mid, vars_z3)
            xn = next_cond_z3(*vars_z3)
            b_now = b_z3_expr(vars_z3, coeff_vals)

            if y0 is not None:
                ce = z3.SolverFor("QF_NRA")
                ce.add(_and_all_z3([dom, y0, b_now > 0]))
                ce_chk = ce.check()
                if ce_chk == z3.sat:
                    solver.add(b_z3(point_from_z3_model(ce.model())) <= 0)
                    continue
                if ce_chk == z3.unknown:
                    return {"success": False, "kind": "smt", "reason": "z3 verifier unknown on Y0", "iterations": it, "last_coefficients": coeff_vals}

            if y1 is not None:
                ce = z3.SolverFor("QF_NRA")
                ce.add(_and_all_z3([dom, y1, b_now < case.margin]))
                ce_chk = ce.check()
                if ce_chk == z3.sat:
                    solver.add(b_z3(point_from_z3_model(ce.model())) >= case.margin)
                    continue
                if ce_chk == z3.unknown:
                    return {"success": False, "kind": "smt", "reason": "z3 verifier unknown on Y1", "iterations": it, "last_coefficients": coeff_vals}

            if y is not None and y1 is not None:
                y_next = _region_union_cond_z3(case, sigma0 + sigma1 + sigma_mid, xn)
                ce = z3.SolverFor("QF_NRA")
                monotonicity_conds = [
                    dom,
                    y,
                    z3.Not(y1),
                    domain_cond_z3(*xn),
                    b_now <= 0,
                    b_z3_expr(xn, coeff_vals) > 0,
                ]
                if y_next is not None:
                    monotonicity_conds.append(y_next)
                ce.add(_and_all_z3(monotonicity_conds))
                ce_chk = ce.check()
                if ce_chk == z3.sat:
                    x_ce = point_from_z3_model(ce.model())
                    solver.add(z3.Implies(b_z3(x_ce) <= 0, b_z3(case.next_num(*x_ce)) <= 0))
                    continue
                if ce_chk == z3.unknown:
                    return {"success": False, "kind": "smt", "reason": "z3 verifier unknown on monotonicity", "iterations": it, "last_coefficients": coeff_vals}

            return {
                "success": True,
                "kind": "smt",
                "coefficients": coeff_vals,
                "verified": "z3",
                "iterations": it,
            }

        return {
            "success": False,
            "kind": "smt",
            "reason": "reached max_iter without a z3-verified barrier",
            "iterations": max_iter,
            "last_coefficients": last_coeff_vals,
        }

    if dreal is None and case.verify_with_dreal:
        # Still check whether the sampled constraints are satisfiable, but do not
        # report a formal certificate without the continuous verifier.
        if solver.check() != z3.sat:
            return {"success": False, "kind": "smt", "reason": "sampled z3 synthesis unsat"}
        model = solver.model()
        coeff_vals = [float(model.evaluate(c, model_completion=True).as_decimal(20).replace("?", "")) for c in coeffs]
        return {
            "success": False,
            "kind": "smt",
            "coefficients": coeff_vals,
            "reason": "dReal unavailable; sampled z3 candidate is not a formal certificate",
        }

    if not case.verify_with_dreal:
        if solver.check() != z3.sat:
            return {"success": False, "kind": "smt", "reason": "sampled z3 synthesis unsat"}
        model = solver.model()
        coeff_vals = [float(model.evaluate(c, model_completion=True).as_decimal(20).replace("?", "")) for c in coeffs]
        return {"success": True, "kind": "smt", "coefficients": coeff_vals, "verified": "samples_only"}

    vars_ = _make_dreal_vars(case.dim)

    def model_point(model: object) -> Point:
        return tuple(float(model[v].mid()) for v in vars_)

    # PT-style CEGIS: Z3 synthesizes candidates over sampled/CE points; dReal
    # searches the continuous domain for counterexamples and feeds them back.
    last_coeff_vals: list[float] | None = None
    for it in range(max_iter):
        if it == 0 or (it + 1) % 100 == 0:
            print(f"[ST-CEGIS] {case.name}: iteration {it + 1}/{max_iter}", flush=True)
        if solver.check() != z3.sat:
            return {
                "success": False,
                "kind": "smt",
                "reason": "z3 synthesis unsat under accumulated sampled/CE constraints",
                "iterations": it,
                "last_coefficients": last_coeff_vals,
            }

        model = solver.model()
        coeff_vals = [float(model.evaluate(c, model_completion=True).as_decimal(20).replace("?", "")) for c in coeffs]
        last_coeff_vals = coeff_vals

        if case.domain_cond_z3 is not None and case.label_cond_z3 is not None:
            vars_static = tuple(z3.Real(f"x{i}_static") for i in range(case.dim))
            dom_static = case.domain_cond_z3(*vars_static)
            b_static = b_z3_expr(vars_static, coeff_vals)

            y0_static = _region_union_cond_z3(case, sigma0, vars_static)
            if y0_static is not None:
                ce = z3.SolverFor("QF_NRA")
                ce.add(_and_all_z3([dom_static, y0_static, b_static > 0]))
                ce_chk = ce.check()
                if ce_chk == z3.sat:
                    x_ce = tuple(z3_value_to_float(ce.model().evaluate(v, model_completion=True)) for v in vars_static)
                    solver.add(b_z3(x_ce) <= 0)
                    continue
                if ce_chk == z3.unknown:
                    return {"success": False, "kind": "smt", "reason": "z3 static verifier unknown on Y0", "iterations": it, "last_coefficients": coeff_vals}

            y1_static = _region_union_cond_z3(case, sigma1, vars_static)
            if y1_static is not None:
                ce = z3.SolverFor("QF_NRA")
                ce.add(_and_all_z3([dom_static, y1_static, b_static < case.margin]))
                ce_chk = ce.check()
                if ce_chk == z3.sat:
                    x_ce = tuple(z3_value_to_float(ce.model().evaluate(v, model_completion=True)) for v in vars_static)
                    solver.add(b_z3(x_ce) >= case.margin)
                    continue
                if ce_chk == z3.unknown:
                    return {"success": False, "kind": "smt", "reason": "z3 static verifier unknown on Y1", "iterations": it, "last_coefficients": coeff_vals}

        ctx = dreal.Context()
        ctx.config.precision = case.dreal_precision
        ctx.SetLogic(dreal.Logic.QF_NRA)
        for i, v in enumerate(vars_):
            lo = min(x[i] for x in case.sample_points)
            hi = max(x[i] for x in case.sample_points)
            ctx.DeclareVariable(v, lo, hi)

        def b_dreal(xs: tuple[object, ...]):
            return sum(coeff_vals[i] * case.template_terms_dreal[i](*xs) for i in range(len(coeff_vals)))

        dom = case.domain_cond(*vars_)
        y0 = _region_union_cond(case, sigma0, vars_)
        y1 = _region_union_cond(case, sigma1, vars_)
        y = _region_union_cond(case, sigma0 + sigma1 + sigma_mid, vars_)
        xn = case.next_cond(*vars_)

        ce_added = False
        if y0 is not None and case.domain_cond_z3 is None:
            ctx.Push(1)
            ctx.Assert(_and_all([dom, y0, b_dreal(vars_) > 0]))
            if it == 0 or (it + 1) % 100 == 0:
                print(f"[ST-dReal] {case.name}: checking Y0", flush=True)
            ce_model = ctx.CheckSat()
            if ce_model is not None:
                x_ce = model_point(ce_model)
                solver.add(b_z3(x_ce) <= 0)
                ce_added = True
            ctx.Pop(1)
            if ce_added:
                continue

        if y1 is not None and case.domain_cond_z3 is None:
            ctx.Push(1)
            # Synthesis uses B >= margin on the target set, so verification must
            # refute B < margin, not merely B <= 0.
            ctx.Assert(_and_all([dom, y1, b_dreal(vars_) < case.margin]))
            if it == 0 or (it + 1) % 100 == 0:
                print(f"[ST-dReal] {case.name}: checking Y1", flush=True)
            ce_model = ctx.CheckSat()
            if ce_model is not None:
                x_ce = model_point(ce_model)
                solver.add(b_z3(x_ce) >= case.margin)
                ce_added = True
            ctx.Pop(1)
            if ce_added:
                continue

        if y is not None and y1 is not None:
            y_next = _region_union_cond(case, sigma0 + sigma1 + sigma_mid, xn)
            monotonicity_conds = [dom, y, dreal.Not(y1), case.domain_cond(*xn), b_dreal(vars_) <= 0, b_dreal(xn) > 0]
            if y_next is not None:
                monotonicity_conds.append(y_next)
            ctx.Push(1)
            ctx.Assert(_and_all(monotonicity_conds))
            if it == 0 or (it + 1) % 100 == 0:
                print(f"[ST-dReal] {case.name}: checking invariance", flush=True)
            ce_model = ctx.CheckSat()
            if ce_model is not None:
                x_ce = model_point(ce_model)
                xn_ce = case.next_num(*x_ce)
                solver.add(z3.Implies(b_z3(x_ce) <= 0, b_z3(xn_ce) <= 0))
                ce_added = True
            ctx.Pop(1)
            if ce_added:
                continue

        return {
            "success": True,
            "kind": "smt",
            "coefficients": coeff_vals,
            "verified": "z3+dreal",
            "iterations": it,
        }

    return {
        "success": False,
        "kind": "smt",
        "reason": "reached max_iter without a dReal-verified barrier",
        "iterations": max_iter,
        "last_coefficients": last_coeff_vals,
    }


def analyze_case(case: CaseModel, max_iter: int = 50) -> dict:
    automaton = case.automaton
    overall_success = True
    accepting_reports: list[dict] = []

    for q_acc in automaton.accepting_states:
        path_reports: list[dict] = []
        cycle_reports: list[dict] = []

        for p in ppath(automaton, q_acc):
            triplets = pf3(p)
            if not triplets:
                edge_report = None
                resolved = False
                if len(p) == 2:
                    edge_labels = automaton.labels_between(p[0], p[1])
                    resolved = _is_empty_region(case, edge_labels)
                    edge_report = {
                        "edge": [p[0], p[1]],
                        "labels": [label_to_str(a) for a in edge_labels],
                        "resolved": resolved,
                        "mode": "edge-empty" if resolved else "direct-edge",
                        "reason": (
                            "edge label region is empty"
                            if resolved
                            else (
                                "dReal unavailable; edge emptiness not formally discharged"
                                if dreal is None and edge_labels
                                else "direct accepting edge has nonempty label region"
                            )
                        ),
                    }
                if not resolved:
                    overall_success = False
                path_reports.append(
                    {
                        "path": p,
                        "pf3": [],
                        "resolved": resolved,
                        "reason": "no length-3 subpath under strict Wong PF^3 framework",
                        "edge": edge_report,
                    }
                )
                continue

            solved = False
            triplet_reports = []
            for tri in triplets:
                q0, q1, q2 = tri
                sigma0 = automaton.labels_between(q0, q1)
                sigma1 = automaton.labels_between(q1, q2)
                sigma_mid = automaton.labels_between(q1, q1) if automaton.has_self_loop(q1) else []

                if not sigma0 or not sigma1:
                    trip = {
                        "triplet": tri,
                        "resolved": True,
                        "mode": "vacuous",
                        "reason": "empty edge label set",
                    }
                    solved = True
                elif _is_empty_intersection(case, sigma0, sigma1):
                    trip = {
                        "triplet": tri,
                        "resolved": True,
                        "mode": "lemma6",
                        "reason": "Y0 ∩ Y1 is empty",
                    }
                    solved = True
                else:
                    smt = synthesize_discrete_barrier(case, sigma0, sigma1, sigma_mid, max_iter=max_iter)
                    trip = {
                        "triplet": tri,
                        "resolved": bool(smt.get("success")),
                        "mode": "corollary2" if sigma_mid else "corollary1",
                        "sigma0": [label_to_str(a) for a in sigma0],
                        "sigma1": [label_to_str(a) for a in sigma1],
                        "sigma_mid": [label_to_str(a) for a in sigma_mid],
                        "backend": smt,
                    }
                    solved = solved or bool(smt.get("success"))
                triplet_reports.append(trip)

            if not solved:
                overall_success = False
            path_reports.append({"path": p, "pf3": triplets, "resolved": solved, "triplets": triplet_reports})

        for c in pcyc(automaton, q_acc):
            cycle_reports.append(
                {
                    "path": c,
                    "pf3": pf3(c),
                    "resolved": True,
                    "mode": "not-required",
                    "reason": "path-blocking condition proves accepting states unreachable",
                }
            )

        accepting_reports.append(
            {
                "accepting_state": q_acc,
                "ppath": path_reports,
                "pcyc": cycle_reports,
            }
        )

    backend_iterations = 0
    for acc in accepting_reports:
        for pr in acc.get("ppath", []):
            for tr in pr.get("triplets", []):
                it = tr.get("backend", {}).get("iterations")
                if isinstance(it, int):
                    backend_iterations = max(backend_iterations, it)

    return {
        "success": overall_success,
        "status": "proved" if overall_success else "inconclusive",
        "iterations": backend_iterations,
        "strict_wong_pf3": True,
        "backend": f"SMT(z3+{case.formal_backend} discrete-time adaptation)",
        "accepting_reports": accepting_reports,
    }
