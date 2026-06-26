# ex3/NNT monotone-ReLU formal results

Date: 2026-06-20
Remote workdir: `/root/ltl_repro_work_20260618/project-new`
Local sync: `project-new/results/legal_search/ex3_nnt_monotone_relu_z3/`

This directory supersedes the narrower `ex3_nnt_hinge_relu_z3` diagnostic run.  The template here is a more general raw-input one-hidden-layer positive ReLU network:

```text
B(x1,x2) = bias + sum_i ai * ReLU(ti - (wi1*x1 + wi2*x2))
wi1, wi2 >= 0, wi1 + wi2 = 1, ai >= 0
```

It covers the axis-hinge certificate family but does not fix the hidden units to `tau-x1` or `tau-x2`.  Each hidden unit learns/selects a positive weighted average of the two raw temperatures.  Inputs are exactly `(x1,x2)`; no AP/VF indicator, no region label, and no PT-style hand-coded certificate is used.

Formal verifier: Z3 over real arithmetic with ReLU encoded by `If`.  A run is accepted only when Z3 proves UNSAT for the negated universal obligations:

1. `B(x) >= 0` on `X=[20,34]^2`,
2. `B(x) >= B(f(x))` on the non-accepting part of the box,
3. `B(x) >= B(f(x)) + epsilon` on `VF=[20,26]^2`.

Command pattern:

```bash
python src/ex3/NNT/main.py \
  --template monotone-relu --hidden-dim 3 --verify-backend z3 \
  --seed <0|1|2> --epsilon 1e-3 --epochs 200 --lr 2e-3 \
  --grid-step 1.0 --max-iter 20 --z3-timeout-ms 180000 \
  --out results/legal_search/ex3_nnt_monotone_relu_z3/seed<k>_h3.json
```

Results:

| seed | success | elapsed_sec | verifier | hidden_dim | thresholds | epsilon |
|---:|:---:|---:|:---:|---:|:---|---:|
| 0 | true | 97.8239 | z3 | 3 | [26.451349, 26.788376, 27.075943] | 0.001 |
| 1 | true | 88.3247 | z3 | 3 | [27.970013, 27.101843, 27.466154] | 0.001 |
| 2 | true | 88.7565 | z3 | 3 | [27.297117, 28.490351, 27.626747] | 0.001 |

Proof evidence is in each `seed*_h3.log`:

```text
✓ nonnegativity verified by Z3 UNSAT
✓ non-increasing verified by Z3 UNSAT
✓ strict-decrease verified by Z3 UNSAT
```

Smoke tests are saved as `seed*_h3_test.json`; they are only auxiliary checks, not the proof.
