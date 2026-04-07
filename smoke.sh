#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[smoke] 1) Syntax check"
python -m compileall -q src

echo "[smoke] 2) Static sanity checks"
python tools/check_sanity.py

echo "[smoke] 3) ex1"
python -u src/ex1/PT/main.py --out smoke_ex1_pt.json
python -u src/ex1/NNT/main.py --out smoke_ex1_nnt.json --max-iter 1 --epochs 1
python -u src/ex1/CC/main.py --out smoke_ex1_cc.json
python -u src/ex1/NCC/main.py --out smoke_ex1_ncc.json --epochs 1 --grid-step 0.01 --eta 0.01

echo "[smoke] 4) ex2"
python -u src/ex2/PT/main.py --max-iter 20 --grid-step 0.2 --dreal-precision 1e-4 --z3-timeout-ms 5000 --out smoke_ex2_pt.json
python -u src/ex2/PT/test.py
python -u src/ex2/NNT/main.py --out smoke_ex2_nnt.json --max-iter 1 --epochs 1
python -u src/ex2/NNT/test.py
python -u src/ex2/CC/main.py --out smoke_ex2_cc.json --max-iter 1
python -u src/ex2/NCC/main.py --out smoke_ex2_ncc.json --epochs 1 --grid-step 0.01 --eta 0.01

echo "[smoke] 5) ex3"
python -u src/ex3/PT/main.py --qi 1 --qj 1 --out smoke_ex3_pt.json
python -u src/ex3/NNT/main.py --out smoke_ex3_nnt.json
python -u src/ex3/CC/main.py --mode main --out smoke_ex3_cc.json
python -u src/ex3/NCC/main.py --mode main --out smoke_ex3_ncc.json --epochs 1 --grid-step 0.01 --eta 0.01

echo "[smoke] Done."
