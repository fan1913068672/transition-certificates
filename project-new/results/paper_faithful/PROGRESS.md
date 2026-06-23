# Paper-faithful CC/NCC progress

## Confirmed

- CC ex1: success, dReal formal verification, 0.6724s.
  - Result: `project-new/results/paper_faithful/cc_ex1/seed0.json`
  - Remote test: `[PASS] ex1/CC checks: 15/15`.

## Important caveat

The previous ex2 CC/NCC fast results under guarded-transition / interval-overapprox semantics are not paper-faithful baselines and should not be used as CC/NCC paper reproduction.

## Running / pending

- NCC ex1 paper-faithful needs restart: previous nohup command had shell quoting error and did not run.
- CC ex2 paper-faithful needs restart/monitor: previous process disappeared; max-iter=1 smoke timed out at 300s before producing a candidate, consistent with original paper-level difficulty.
