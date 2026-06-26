from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ex3/CC result and test report contracts.")
    parser.add_argument("--result", required=True, help="Path to the ex3/CC result JSON.")
    parser.add_argument("--test-report", required=True, help="Path to the ex3/CC test report JSON.")
    parser.add_argument("--command", default="", help="Command used to produce the result.")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "results" / "cc_ex3_summary_manifest.json"))
    args = parser.parse_args()

    result_path = Path(args.result)
    test_path = Path(args.test_report)
    result = _load_json(result_path)
    test_report = _load_json(test_path)

    result_exists = result is not None
    test_report_exists = test_report is not None
    result_success = bool(result.get("success")) if isinstance(result, dict) else False
    test_passed = bool(test_report.get("passed")) if isinstance(test_report, dict) else False
    coefficients = result.get("coefficients") if isinstance(result, dict) else None
    first_failed_check = None
    if isinstance(test_report, dict):
        for check in test_report.get("checks", []):
            if isinstance(check, dict) and not check.get("passed"):
                first_failed_check = check.get("name")
                break

    if not result_exists or not test_report_exists:
        classification = "artifact-packaging-gap"
    elif args.exit_code != 0:
        classification = "implementation-bug"
    elif result_success and test_passed and coefficients:
        classification = "evidence-signal"
    elif test_passed:
        classification = "weak-signal"
    else:
        classification = "repair-needed"

    manifest = {
        "experiment_id": "cc_ex3_result_contract_repair_p2",
        "command": args.command,
        "exit_code": args.exit_code,
        "result_path": str(result_path),
        "test_report_path": str(test_path),
        "result_exists": result_exists,
        "test_report_exists": test_report_exists,
        "result_success": result_success,
        "test_passed": test_passed,
        "first_failed_check": first_failed_check,
        "backend": result.get("backend") if isinstance(result, dict) else None,
        "mode": result.get("mode") if isinstance(result, dict) else None,
        "coefficients_present": bool(coefficients),
        "elapsed_sec": result.get("elapsed_sec") if isinstance(result, dict) else None,
        "result_sha256": _sha256(result_path),
        "test_report_sha256": _sha256(test_path),
        "failure_classification": classification,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0 if classification in {"evidence-signal", "weak-signal"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
