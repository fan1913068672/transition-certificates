from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a bounded cc_ex3 diagnostic manifest.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--wrapper", default="")
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    summary_path = Path(args.summary).resolve()
    wrapper_path = Path(args.wrapper).resolve() if args.wrapper else None
    summary = _read_json(summary_path)

    manifest = {
        "experiment_id": "cc_ex3_result_contract_repair_p2",
        "status": "diagnostic-manifest-only",
        "failure_classification": (
            summary.get("failure_classification")
            if isinstance(summary, dict)
            else "artifact-packaging-gap"
        ),
        "signal_level": (
            "weak-signal"
            if isinstance(summary, dict) and summary.get("test_passed") is True
            else "repair-needed"
        ),
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "repo_commit": _git_commit(repo_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timeout_sec": args.timeout_sec,
        "summary_manifest": str(summary_path),
        "summary_manifest_sha256": _sha256(summary_path),
        "wrapper": str(wrapper_path) if wrapper_path else None,
        "wrapper_sha256": _sha256(wrapper_path) if wrapper_path else None,
        "result_contract": {
            "result_exists": summary.get("result_exists") if isinstance(summary, dict) else None,
            "test_report_exists": summary.get("test_report_exists") if isinstance(summary, dict) else None,
            "result_success": summary.get("result_success") if isinstance(summary, dict) else None,
            "test_passed": summary.get("test_passed") if isinstance(summary, dict) else None,
            "coefficients_present": summary.get("coefficients_present") if isinstance(summary, dict) else None,
            "first_failed_check": summary.get("first_failed_check") if isinstance(summary, dict) else None,
        },
        "claim_boundary": (
            "This diagnostic can support only evidence-signal or weak-signal. "
            "It does not approve a server run and does not prove the paper theorem."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
