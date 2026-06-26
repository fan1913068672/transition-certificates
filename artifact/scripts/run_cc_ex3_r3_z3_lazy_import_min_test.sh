#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/root/research_runs/remote-ltl-cc-ex3-r3-z3-lazy-import}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="$RUN_ROOT/logs"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
RESULT_JSON="$ARTIFACT_DIR/cc_ex3_r3_z3_lazy_import_result.json"
MANIFEST_JSON="$ARTIFACT_DIR/cc_ex3_r3_z3_lazy_import_manifest.json"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

{
  date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ"
  python3 --version
  python3 - <<'PY'
import importlib.util
for name in ["torch", "numpy", "sympy", "z3", "dreal"]:
    print(f"{name}_available={importlib.util.find_spec(name) is not None}")
PY
} > "$LOG_DIR/env.log" 2>&1

COMMAND=(
  python3 "$REPO_ROOT/artifact/src/ex3/NNT/main.py"
  --verify-backend z3
  --z3-timeout-ms 30000
  --max-iter 1
  --epochs 1
  --seed 0
  --out "$RESULT_JSON"
)

set +e
"${COMMAND[@]}" > "$LOG_DIR/run.log" 2>&1
code=$?
set -e
printf 'EXIT:%s\n' "$code" > "$LOG_DIR/exit.code"

python3 - "$REPO_ROOT" "$RUN_ROOT" "$RESULT_JSON" "$MANIFEST_JSON" "$code" "${COMMAND[*]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

repo_root = pathlib.Path(sys.argv[1]).resolve()
run_root = pathlib.Path(sys.argv[2]).resolve()
result_json = pathlib.Path(sys.argv[3]).resolve()
manifest_json = pathlib.Path(sys.argv[4]).resolve()
exit_code = int(sys.argv[5])
command = sys.argv[6]

def sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

result = None
if result_json.exists():
    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        result = None

if exit_code == 0 and isinstance(result, dict):
    signal = "evidence-signal" if result.get("success") is True else "negative-signal"
    failure_classification = None if result.get("success") is True else "paper-claim-gap"
elif result_json.exists():
    signal = "repair-needed"
    failure_classification = "solver-or-certificate-diagnostic-failed"
else:
    signal = "repair-needed"
    failure_classification = "artifact-packaging-gap"

manifest = {
    "experiment_id": "cc_ex3_r3_z3_lazy_import_min_test",
    "status": "complete" if exit_code == 0 else "repair-needed",
    "signal_label": signal,
    "failure_classification": failure_classification,
    "exit_code": exit_code,
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "repo_root": str(repo_root),
    "repo_commit": git_commit(),
    "run_root": str(run_root),
    "command": command,
    "result_json": str(result_json),
    "result_json_sha256": sha256(result_json),
    "log_path": str(run_root / "logs" / "run.log"),
    "env_log_path": str(run_root / "logs" / "env.log"),
    "claim_boundary": (
        "This is a bounded z3-only diagnostic after the lazy dReal import repair. "
        "It is signal-only and cannot be reported as theorem verification, superiority, "
        "statistical stability, or a reviewer-facing main claim."
    ),
}
manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {manifest_json}")
PY

exit "$code"
