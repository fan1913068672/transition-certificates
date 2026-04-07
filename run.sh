#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TIMEOUT_SEC=3600

run_cmd() {
  local name="$1"
  shift
  echo "[run] >>> ${name}"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${TIMEOUT_SEC}" "$@"
  else
    "$@"
  fi
  echo "[run] <<< ${name}"
}

echo "[run] 1) Syntax check"
run_cmd "compileall" python -m compileall -q src
run_cmd "sanity-check" python tools/check_sanity.py

echo "[run] 2) ex1 full"
run_cmd "ex1 PT"  python -u src/ex1/PT/main.py --out run_ex1_pt.json
run_cmd "ex1 NNT" python -u src/ex1/NNT/main.py --out run_ex1_nnt.json
run_cmd "ex1 CC"  python -u src/ex1/CC/main.py --out run_ex1_cc.json
run_cmd "ex1 NCC" python -u src/ex1/NCC/main.py --out run_ex1_ncc.json

echo "[run] 3) ex2 full"
run_cmd "ex2 PT"  python -u src/ex2/PT/main.py --max-iter 1000 --grid-step 0.1 --dreal-precision 1e-4 --z3-timeout-ms 0 --out run_ex2_pt.json
run_cmd "ex2 PT test" python -u src/ex2/PT/test.py
run_cmd "ex2 NNT" python -u src/ex2/NNT/main.py --out run_ex2_nnt.json
run_cmd "ex2 NNT test" python -u src/ex2/NNT/test.py
run_cmd "ex2 CC"  python -u src/ex2/CC/main.py --out run_ex2_cc.json
run_cmd "ex2 NCC" python -u src/ex2/NCC/main.py --out run_ex2_ncc.json

echo "[run] 4) ex3 full"
run_cmd "ex3 PT"  python -u src/ex3/PT/main.py --qi 1 --qj 1 --out run_ex3_pt.json
run_cmd "ex3 PT test" python -u src/ex3/PT/test.py
run_cmd "ex3 PT safety" python -u src/ex3/PT/safety.py
run_cmd "ex3 NNT" python -u src/ex3/NNT/main.py --out run_ex3_nnt.json
run_cmd "ex3 CC"  python -u src/ex3/CC/main.py --mode main --out run_ex3_cc.json
run_cmd "ex3 NCC" python -u src/ex3/NCC/main.py --mode main --out run_ex3_ncc.json

echo "[run] All done."
