from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any


def load_local_main(test_file: str):
    test_path = Path(test_file).resolve()
    main_path = test_path.parent / "main.py"
    spec = importlib.util.spec_from_file_location(f"{test_path.parent.name}_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=str, default="test_result.json", help="output JSON path")
    p.add_argument("--result", type=str, default="", help="optional synthesis result JSON to validate")
    p.add_argument("--x0-samples", type=int, default=100, help="number of initial points for sampled trajectory check")
    p.add_argument("--sim-steps", type=int, default=300, help="simulation horizon for sampled trajectory check")
    return p


def make_report(example: str, method: str) -> dict[str, Any]:
    return {
        "example": example,
        "method": method,
        "passed": True,
        "checks": [],
        "errors": [],
    }


def add_check(report: dict[str, Any], name: str, passed: bool, details: str = "") -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "details": details})
    if not passed:
        report["passed"] = False
        if details:
            report["errors"].append(f"{name}: {details}")
        else:
            report["errors"].append(name)


def add_exception(report: dict[str, Any], err: BaseException) -> None:
    report["passed"] = False
    report["errors"].append(f"{type(err).__name__}: {err}")
    report["traceback"] = traceback.format_exc()


def save_and_print(report: dict[str, Any], out_arg: str, test_file: str) -> int:
    out_path = Path(out_arg)
    if not out_path.is_absolute():
        out_path = Path(test_file).resolve().parent / out_path
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total = len(report["checks"])
    passed = sum(1 for c in report["checks"] if c["passed"])
    status = "PASS" if report["passed"] else "FAIL"
    print(f"[{status}] {report['example']}/{report['method']} checks: {passed}/{total}")
    print(f"Saved test report to: {out_path}")
    return 0 if report["passed"] else 1


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [(a + b) / 2.0]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def check_sampled_initial_points(report: dict[str, Any], module: Any, example: str, count: int = 100, steps: int = 300) -> None:
    """Numerically check the benchmark property from sampled initial points.

    This is a sanity/effectiveness check for generated results, not a replacement
    for the formal certificate verification.
    """
    import math

    count = max(1, int(count))
    steps = max(1, int(steps))
    pi = getattr(module, "PI", math.pi) if module is not None else math.pi

    def f1(x: float) -> float:
        if module is not None and hasattr(module, "f_num"):
            y = module.f_num(x)
            return float(y[0] if isinstance(y, (tuple, list)) else y)
        if module is not None and hasattr(module, "f_m"):
            y = module.f_m(x)
            return float(y[0] if isinstance(y, (tuple, list)) else y)
        if module is not None and hasattr(module, "f"):
            return float(module.f(x))
        return float(x + 0.1 * 0.01 + 0.1 * 0.0006 * math.sin(-x) - 0.532 * x * x + 1.69)

    def f2(x1: float, x2: float) -> tuple[float, float]:
        if module is not None and hasattr(module, "f_num"):
            y = module.f_num(x1, x2)
        elif module is not None and hasattr(module, "f_m"):
            y = module.f_m(x1, x2)
        elif module is not None and hasattr(module, "f"):
            y = module.f(x1, x2)
        else:
            if example == "ex3":
                alpha, theta, mu, th, te = 0.004, 0.01, 0.15, 40.0, 0.0

                def u(x: float) -> float:
                    return 0.59 - 0.011 * x

                y = (
                    (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te,
                    alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te,
                )
            else:
                ts, omega, k = 0.1, 0.01, 0.0006
                y = (
                    x1 + ts * omega + 1.69 + ts * k * math.sin(x2 - x1) - 0.532 * x1 * x1,
                    x2 + ts * omega + 1.69 + ts * k * math.sin(x1 - x2) - 0.532 * x2 * x2,
                )
        return float(y[0]), float(y[1])

    def in_domain1(x: float) -> bool:
        if module is not None and hasattr(module, "in_x"):
            return bool(module.in_x(x))
        if module is not None and hasattr(module, "in_state_space"):
            return bool(module.in_state_space(x))
        return 0.0 <= x <= 2 * pi

    def unsafe1(x: float) -> bool:
        if module is not None and hasattr(module, "in_xu"):
            return bool(module.in_xu(x))
        if module is not None and hasattr(module, "in_unsafe"):
            return bool(module.in_unsafe(x))
        if module is not None and hasattr(module, "in_unsafe_set_numeric"):
            return bool(module.in_unsafe_set_numeric(x))
        return 7 * pi / 9 <= x <= 8 * pi / 9

    def in_domain2(x1: float, x2: float) -> bool:
        if module is not None and hasattr(module, "in_x"):
            return bool(module.in_x(x1, x2))
        return 0.0 <= x1 <= 8 * pi / 9 and 0.0 <= x2 <= 8 * pi / 9

    def unsafe2(x1: float, x2: float) -> bool:
        if module is not None and hasattr(module, "in_xu"):
            return bool(module.in_xu(x1, x2))
        if module is not None and hasattr(module, "in_unsafe"):
            return bool(module.in_unsafe(x1, x2))
        if module is not None and hasattr(module, "In_Unsafe"):
            return bool(module.In_Unsafe(x1, x2))
        return (5 * pi / 6 <= x1 <= 8 * pi / 9) or (5 * pi / 6 <= x2 <= 8 * pi / 9)

    def in_domain3(x1: float, x2: float) -> bool:
        if module is not None and hasattr(module, "in_x"):
            return bool(module.in_x(x1, x2))
        return 20.0 <= x1 <= 34.0 and 20.0 <= x2 <= 34.0

    def vf3(x1: float, x2: float) -> bool:
        if module is not None and hasattr(module, "in_vf"):
            return bool(module.in_vf(x1, x2))
        if module is not None and hasattr(module, "In_VF"):
            return bool(module.In_VF(x1, x2))
        return 20.0 <= x1 <= 26.0 and 20.0 <= x2 <= 26.0

    if example == "ex1":
        xs = _linspace(4 * pi / 9, 5 * pi / 9, count)
        bad = []
        for x0 in xs:
            x = x0
            ok = True
            for _ in range(steps):
                if not math.isfinite(x) or not in_domain1(x) or unsafe1(x):
                    ok = False
                    break
                x = f1(x)
            if ok and (not math.isfinite(x) or not in_domain1(x) or unsafe1(x)):
                ok = False
            if not ok:
                bad.append(x0)
                if len(bad) >= 3:
                    break
        add_check(report, f"sampled X0 safety ({count} points)", not bad, f"counterexample initial points: {bad}")
        return

    if example == "ex2":
        side = max(1, int(math.ceil(math.sqrt(count))))
        vals = _linspace(0.0, pi / 9, side)
        pts = [(x1, x2) for x1 in vals for x2 in vals][:count]
        bad = []
        for x10, x20 in pts:
            x1, x2 = x10, x20
            ok = True
            for _ in range(steps):
                if not (math.isfinite(x1) and math.isfinite(x2)) or not in_domain2(x1, x2) or unsafe2(x1, x2):
                    ok = False
                    break
                x1, x2 = f2(x1, x2)
            if ok and (not (math.isfinite(x1) and math.isfinite(x2)) or not in_domain2(x1, x2) or unsafe2(x1, x2)):
                ok = False
            if not ok:
                bad.append((x10, x20))
                if len(bad) >= 3:
                    break
        add_check(report, f"sampled X0 safety ({len(pts)} points)", not bad, f"counterexample initial points: {bad}")
        return

    if example == "ex3":
        side = max(1, int(math.ceil(math.sqrt(count))))
        vals = _linspace(21.0, 24.0, side)
        pts = [(x1, x2) for x1 in vals for x2 in vals][:count]
        bad = []
        for x10, x20 in pts:
            x1, x2 = x10, x20
            vf_flags = []
            ok = True
            for _ in range(steps):
                if not (math.isfinite(x1) and math.isfinite(x2)) or not in_domain3(x1, x2):
                    ok = False
                    break
                vf_flags.append(vf3(x1, x2))
                x1, x2 = f2(x1, x2)
            # For a => FG not b, require a suffix with no VF visits.
            tail = max(1, min(50, len(vf_flags)))
            if (not ok) or any(vf_flags[-tail:]):
                bad.append((x10, x20))
                if len(bad) >= 3:
                    break
        add_check(report, f"sampled X0 persistence ({len(pts)} points)", not bad, f"non-stabilizing initial points: {bad}")
        return

    add_check(report, "sampled X0 check", False, f"unknown example {example}")
