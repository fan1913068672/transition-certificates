#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/root/research_runs/remote-ltl-cc-ex3-r2-repair}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="$RUN_ROOT/logs"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

{
  python3 --version
  python3 -m pip freeze
  python3 - <<'PY'
try:
    import z3
    print("z3_version=" + ".".join(map(str, z3.get_version())))
except Exception as exc:
    print("z3_version_error=" + repr(exc))
PY
} > "$LOG_DIR/env.log" 2>&1

SUMMARY="$ARTIFACT_DIR/cc_ex3_summary_manifest.json"

set +e
python3 "$REPO_ROOT/artifact/scripts/cc_ex3_summary_writer.py" \
  --result "$REPO_ROOT/artifact/src/ex3/CC/res_cc_ex3_gfb.json" \
  --test-report "$REPO_ROOT/artifact/src/ex3/CC/test_result_100x0_res_cc_ex3_gfb.json" \
  --command "python artifact/src/ex3/CC/main.py; python artifact/src/ex3/CC/test.py" \
  --exit-code 0 \
  --out "$SUMMARY" \
  > "$LOG_DIR/run.log" 2>&1
code=$?
set -e
printf 'EXIT:%s\n' "$code" > "$LOG_DIR/exit.code"

python3 "$REPO_ROOT/artifact/scripts/write_diagnostic_manifest.py" \
  --run-root "$RUN_ROOT" \
  --repo-root "$REPO_ROOT" \
  --summary "$SUMMARY" \
  --wrapper "$REPO_ROOT/artifact/scripts/run_cc_ex3_r2_diagnostic.sh" \
  --output "$ARTIFACT_DIR/diagnostic_manifest.json"

exit "$code"
