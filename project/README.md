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

## 3) Run commands

Run from `project/` root.

### ex1

```bash
python src/ex1/PT/main.py --out res_pt_ex1.json
python src/ex1/PT/test.py

python src/ex1/NNT/main.py
python src/ex1/NNT/test.py

python src/ex1/CC/main.py --out res_cc_ex1.json

python src/ex1/NCC/main.py
# example with explicit NCC params:
# python src/ex1/NCC/main.py --epochs 1500 --xi 0.01 --eta 0.01
```

### ex2

```bash
python src/ex2/PT/main.py --out res_pt_ex2.json

python src/ex2/NNT/main.py
python src/ex2/NNT/test.py

python src/ex2/CC/main.py

python src/ex2/NCC/main.py
# example:
# python src/ex2/NCC/main.py --epochs 2000 --xi 0.01 --eta 0.01
```

### ex3

```bash
python src/ex3/PT/main.py --qi 1 --qj 1 --out res_pt_ex3.json
python src/ex3/PT/test.py
python src/ex3/PT/safety.py

python src/ex3/NNT/main.py
python src/ex3/NNT/test.py

python src/ex3/CC/main.py --mode main
# optional: python src/ex3/CC/main.py --mode closure

python src/ex3/NCC/main.py --mode main
# optional: python src/ex3/NCC/main.py --mode closure
# NCC scripts output `certified` and `theorem_margin` in `res.txt`/`saved_models/model.json`.
```

Result files:
- PT/CC scripts now save a JSON result file by default (or to `--out` path).
- NCC scripts output `res.txt` and `saved_models/model.json`.
- NNT scripts save trained model checkpoints (`*.pth`).
