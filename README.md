# Transition Certificates Project

This folder contains the open-source experiment code for:
- `ex1`: 1D Kuramoto oscillator
- `ex2`: 2D Kuramoto oscillator
- `ex3`: two-room temperature model

Methods:
- `PT`: polynomial transition certificates
- `NNT`: neural transition certificates
- `CC`: closure certificates
- `NCC`: neural closure certificates

## 1) Environment

From `project/`:

```bash
python -m pip install -r requirements.txt
```

Solver dependencies:
- `z3-solver`: required by PT/CC scripts.
- `dreal`: required by scripts that import `dreal` (e.g., ex1 CC/PT, NNT scripts).
- `torch`: required by NNT/NCC scripts.

CC solver policy in this repository:
- `ex2/CC`: uses **Z3 for candidate synthesis** + **dReal for counterexample checking**.
- `ex3/CC`: uses **Z3**.

## 2) Unified file naming

Each method directory now uses:
- `main.py`: main synthesis/verification entry
- `test.py`: test entry (if provided)

Special case:
- `src/ex3/PT/safety.py`: state-safety helper script for ex3 PT.

## 2.1) Result/data files (kept for submission)

Generated artifacts are intentionally kept in this repository for reproducibility:

- JSON outputs under each method folder (e.g., `src/ex2/PT/res_pt_ex2.json`).
- Test reports under each method folder (`test_result.json`).
- Neural checkpoints under `saved_models/` and root-level `*.pth` files.

Recommended output naming:
- smoke runs: `smoke_*.json`
- full runs: `run_*.json`
- paper/main runs: `res_*.json`

## 3) Unified CLI arguments

All `main.py` entrypoints accept a common argument surface:

- `--out`
- `--max-iter`
- `--epochs`
- `--lr`
- `--grid-step`
- `--dreal-precision`
- `--z3-timeout-ms`
- `--seed`
- `--qi`
- `--qj`

Method-specific arguments still exist (for example `--mode` in `ex3/CC` and `ex3/NCC`,
or `--eta --lambda1 --lambda2 --tol` in NCC).

Some common arguments are intentionally ignored by methods that do not use them, to keep a consistent CLI.

## 4) Run commands

Run from `project/` root.

### ex1

```bash
python src/ex1/PT/main.py --out res_pt_ex1.json --max-iter 1000 --grid-step 0.01 --dreal-precision 1e-4 --z3-timeout-ms 0
python src/ex1/PT/test.py

python src/ex1/NNT/main.py --out res_nnt_ex1.json --max-iter 20 --epochs 1000 --lr 0.01
python src/ex1/NNT/test.py

python src/ex1/CC/main.py --out res_cc_ex1.json --max-iter 100000 --grid-step 0.01
python src/ex1/CC/test.py

python src/ex1/NCC/main.py --out res_ncc_ex1.json --epochs 1500 --grid-step 0.01 --eta 0.01
python src/ex1/NCC/test.py
```

### ex2

```bash
python src/ex2/PT/main.py --out res_pt_ex2.json --max-iter 1000 --grid-step 0.1 --dreal-precision 1e-4 --z3-timeout-ms 0
python src/ex2/PT/test.py

python src/ex2/NNT/main.py --out res_nnt_ex2.json --max-iter 2000 --epochs 1000 --lr 0.01
python src/ex2/NNT/test.py

python src/ex2/CC/main.py --out res_cc_ex2.json --max-iter 100000 --grid-step 0.01
python src/ex2/CC/test.py

python src/ex2/NCC/main.py --out res_ncc_ex2.json --epochs 2000 --grid-step 0.01 --eta 0.01
python src/ex2/NCC/test.py
```

### ex3

```bash
python src/ex3/PT/main.py --qi 1 --qj 1 --out res_pt_ex3.json --max-iter 1000
python src/ex3/PT/test.py
python src/ex3/PT/safety.py

python src/ex3/NNT/main.py --out res_nnt_ex3.json --max-iter 1000 --epochs 50 --lr 1e-4
python src/ex3/NNT/test.py

python src/ex3/CC/main.py --mode main --out res_cc_ex3.json  --max-iter 100000 --grid-step 0.01
python src/ex3/CC/test.py

python src/ex3/NCC/main.py --mode main --out res_ncc_ex3.json --epochs 2200 --grid-step 0.01 --eta 0.01
python src/ex3/NCC/test.py --mode main
```

Result files:
- PT/CC/NCC/NNT scripts save a JSON result file to `--out`.
- NCC scripts additionally output `res.txt` and `saved_models/model.json`.
- NNT scripts additionally save trained model checkpoints (`*.pth`).

## 5) Batch scripts

- `smoke.sh`: quick smoke run across ex1/ex2/ex3.
- `run.sh`: full run with 1-hour timeout wrapper per command (if `timeout` is available).

