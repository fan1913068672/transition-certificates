# CC/NCC fix progress (2026-06-20)

Formal baseline fixes/results for `project-new`.

## Root cause found

`ex2` two-dimensional Kuramoto dynamics in code used `-0.532*x_i^2`, while `latest-paper/casestudy.tex` and Murali et al. CC paper use `-0.532*t_s*x_i^2`. This invalidated the previous ex2 formal/comparison runs.

Fixed in all `project-new/src/ex2/*/main.py` methods.

## CC ex2

Verifier: Z3 with a sound interval over-approximation of the sinusoidal term (`|sin| <= 1`). Passing this relation proves the exact sin dynamics.

| seed | result | time |
|---|---:|---:|
| 0 | success | 14.6796s |
| 1 | success | 14.4882s |
| 2 | success | 14.4179s |

Files: `results/fix_cc/res_cc_ex2_ts_seed*.json`; tests pass `16/16`.

## NCC ex2

Implementation changed from single-point PyTorch calls to vectorized training + chunked exhaustive grid check. Transition premise is guarded by `x in X and f(x) in X`. Lipschitz bound updated to `1 + 2*t_s*K = 1.00012` for corrected dynamics.

| seed | train eta | cert eta | result | time |
|---|---:|---:|---:|---:|
| 0 | 0.05 | 0.0470 | certified | 119.6491s |
| 1 | 0.05 | 0.0470 | certified | 121.6096s |
| 2 | 0.05 | 0.0465 | certified | 119.9672s |

Files: `results/fix_ncc/ex2_ncc_ts_vec_seed*.json`; tests pass `12/12`.

## ex3 CC observation

Original Murali D.2 two-room coefficients do not satisfy the strict global Definition 3.3 obligations under direct Z3 checking; Z3 finds a C1 counterexample. This suggests the original artifact relied on implicit reachable-region/sample restrictions, not the strict global obligations used here.

## NCC ex1 probe (2026-06-21)

Remote probes under the guarded transition semantics:

| run | result | key numbers |
|---|---|---|
| seed0, xi=0.01, eta=0.01, lip_reg=0.01, epochs=5000 | not certified | theorem_margin=0.003375, max_lprime=0.053977, min_g2=-0.043977 |
| seed0, xi=0.02, eta=0.05, lip_reg=0.05, epochs=5000 | not certified | theorem_margin=-0.013793, max_lprime=0.162814 |
| seed0, xi=0.02, eta=0.05, lip_reg=0.001, epochs=5000 | not certified | theorem_margin=0.264219, max_lprime=0.212109 |

Diagnosis: ex1/NCC is not blocked mainly by dReal/Z3; the current random mini-batch training does not satisfy the exhaustive finite-grid NCC loss, especially g2.  Blindly increasing epochs or changing eta/lip-reg is unlikely to be enough.  Next implementation step should add hard-mining or full-grid fine-tuning over the finite representatives before applying the Lipschitz theorem check.  This remains within the NCC theorem and does not use sampling-only verification.
