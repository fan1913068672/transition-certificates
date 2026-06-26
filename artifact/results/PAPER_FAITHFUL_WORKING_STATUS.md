# Paper-faithful experiment status (working note)

Scope: CC/NCC baselines must be faithful to the original papers and must pass formal verification/certification. Sampling-only or guarded/shortcut variants are not counted.

Remote workspace: `/root/ltl_repro_work_20260618/project-new`
Remote active tmux sessions checked on 2026-06-20 HDT / 2026-06-21 local:

## Completed faithful probes

| case | method | output | status | elapsed | note |
|---|---|---|---|---:|---|
| ex1 | CC | results/paper_faithful_5h/cc_ex1/seed0.json | not accepted | 1.77s | `--max-iter 500` ended too early; not a true 5h timeout. Superseded by `cc_ex1_long`. |
| ex1 | NCC | results/paper_faithful_5h/ncc_ex1/seed0.json | certified=false | 88.49s | NCC finite-grid/Lipschitz sufficient conditions failed: max_lprime=0.071456 > tol=1e-4 and theorem_margin=L*xi/2-eta=0.036329 > 0. |
| ex2 | NCC | results/paper_faithful_5h/ncc_ex2/seed0.json | certified=false | 213.03s | NCC finite-grid/Lipschitz sufficient conditions failed: max_lprime=0.149700 > tol=1e-4 and theorem_margin=L*xi/2-eta=0.004631 > 0. |
| ex2 | CC | results/paper_faithful_5h/cc_ex2/seed0.json | success=false | 2512.29s | First Z3-LRA candidate synthesis returned no candidate before any CEGIS refinement (`iterations=0`); no formal certificate reproduced. |
| ex1 | NCC | results/paper_faithful_5h/ncc_ex1_eta005_s1/seed1.json | certified=false | 333.26s | Tuned eta=0.05 seed=1 failed: theorem_margin=0.116525 > 0 and max_lprime=0.282423 > tol. |
| ex2 | NCC | results/paper_faithful_5h/ncc_ex2_eta008_s1/seed1.json | certified=false | 654.64s | Tuned eta=0.08 seed=1 made theorem_margin=-0.000677 <= 0 but failed grid condition: max_lprime=0.204667 > tol. |
| ex1 | NCC | results/paper_faithful_5h/ncc_ex1_xi0005_eta0015_s2/seed2.json | certified=false | 891.92s | Tuned xi=0.005 eta=0.015 seed=2 made theorem_margin=-0.002901 <= 0 but failed grid condition: max_lprime=0.089493 > tol. |

## Active faithful long runs

| tmux | case | method | command intent | started |
|---|---|---|---|---|
| pf_cc_ex1_long | ex1 | CC | 5h timeout, `--max-iter 5000000`, dReal precision 1e-4 | Sat Jun 20 22:44 HDT |
| pf_cc_ex2_domain | ex2 | CC | 5h timeout, exact sin+dReal, only successors in X treated as transitions, max_iter=5000 | Sat Jun 20 23:12 HDT |
| pf_cc_ex3_corrected | ex3 | CC | 5h timeout, corrected 4-state state-based NBA for `a0 ∧ GF a1` / `a ∧ GF b` | Sun Jun 21 00:02 HDT |
| pf_ncc_ex3 | ex3 | NCC | 5h timeout, 5000 epochs, xi=0.5, eta=0.2 | Sat Jun 20 22:22 HDT |

## Reporting rule

Only entries with formal `success/certified=true` from the faithful implementation should be reported as successful. Otherwise keep table entry as `Timeout` or explicitly describe as not certified in internal notes.




Implementation note: ex2 CC paper coefficients fail if obligations are imposed on f(x) outside X; this indicates the faithful transition relation must respect the declared state space X, i.e. only successors in X are transitions. The rerun cc_ex2_domain uses exact sin+dReal and this state-space guard.

Implementation note: the old remote `pf_cc_ex3` run used a 2-state `GF b` automaton and did not encode the initial `a0`/`a` guard. That language is not equivalent to the intended negated LTL `a0 ∧ GF a1` (`a ∧ GF b`). The run was stopped and replaced by `pf_cc_ex3_corrected`, whose automaton matches CC Fig. 5 up to state renaming.



