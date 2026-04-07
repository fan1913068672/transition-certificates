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

