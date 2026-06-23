# Paper-faithful CC/NCC 5h run status

Started remote tmux runs on `/root/ltl_repro_work_20260618/project-new` with a 18000s timeout per case. Guarded-transition / interval-overapproximation variants are excluded.

## Smoke
- CC ex1: CLI and dReal path executed; no certificate in 1-iteration smoke.
- CC ex2: default paper-size constraint construction exceeded the 180s smoke cap before entering the CEGIS loop; this is consistent with the heavy full-sample encoding and not an import/CLI error.
- CC ex3: CLI and Z3 path executed; no certificate in 1-iteration smoke.
- NCC ex1/ex2/ex3: CLI, training, exhaustive finite-grid check, and JSON output all executed with coarse smoke grids.

## Completed so far
| Method | Case | Result | Time | Notes |
|---|---|---:|---:|---|
| CC | ex1 | no certificate | 1.77s | faithful CEGIS template without c1=0 shortcut; max_iter=500 exhausted |
| NCC | ex1 | not certified | 88.49s | xi=0.02, eta=0.01; grid loss and Lipschitz margin fail |
| NCC | ex2 | not certified | 213.03s | xi=0.1, eta=0.05; grid monotonicity and Lipschitz margin fail |

## Still running remotely
- `pf_cc_ex2`
- `pf_cc_ex3`
- `pf_ncc_ex3`

Paper tables currently record unavailable CC/NCC certificates as `Timeout`, per reporting rule.
