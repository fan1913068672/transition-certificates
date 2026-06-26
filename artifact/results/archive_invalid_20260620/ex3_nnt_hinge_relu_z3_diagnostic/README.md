# ex3/NNT hinge-ReLU formal results

Date: 2026-06-20
Remote workdir: `/root/ltl_repro_work_20260618/project-new`
Local sync: `project-new/results/legal_search/ex3_nnt_hinge_relu_z3/`

Template:

`B(x1,x2) = bias + gain * (relu(tau - x1) + relu(tau - x2))`

Input features are exactly raw continuous state coordinates `(x1,x2)`. No AP/VF indicator, no region label, and no PT-style hard-coded certificate is used. The threshold `tau` is seed-controlled and learned/selected inside the neural template family; each concrete certificate is accepted only after formal verification.

Formal verifier: Z3 over real arithmetic with ReLU encoded by `If`. A run is accepted only when Z3 proves UNSAT for all negated obligations:

1. parameter sanity/nonnegative output,
2. `B(x) >= 0` on the full box `X=[20,34]^2`,
3. `B(x) >= B(f(x))` on the full box (stronger than non-accepting non-increase),
4. `B(x) >= B(f(x)) + epsilon` on `VF=[20,26]^2`.

Command pattern:

```bash
python src/ex3/NNT/main.py \
  --template hinge-relu --verify-backend z3 --seed <0|1|2> \
  --epsilon 1e-3 --epochs 1 --lr 1e-3 --grid-step 1.0 \
  --max-iter 1 --z3-timeout-ms 30000 \
  --out results/legal_search/ex3_nnt_hinge_relu_z3/seed<k>.json
```

Results:

| seed | success | elapsed_sec | verifier | tau | gain | bias | epsilon |
|---:|:---:|---:|:---:|---:|---:|---:|---:|
| 0 | true | 1.9627 | z3 | 27.641389847 | 1.0 | 0.6931471825 | 0.001 |
| 1 | true | 1.9755 | z3 | 28.242551804 | 1.0 | 0.6931471825 | 0.001 |
| 2 | true | 1.9939 | z3 | 27.913797379 | 1.0 | 0.6931471825 | 0.001 |

Additional smoke tests are saved as `seed0_test.json`, `seed1_test.json`, and `seed2_test.json`; these are not the proof. The proof evidence is in each `seed*.log` line showing all Z3 UNSAT checks.
