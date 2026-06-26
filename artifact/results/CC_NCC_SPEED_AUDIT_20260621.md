# CC/NCC speed audit (2026-06-21)

This note records why the current reproduced CC/NCC runs can be much faster than the timings stated in the original CC/NCC papers.  It is an audit note, not paper text.

## Current confirmed local/remote-synced results

### ex2 / CC

Files: `results/fix_cc/res_cc_ex2_ts_seed{0,1,2}.json`.

| seed | result | elapsed |
|---:|---|---:|
| 0 | success | 14.6796s |
| 1 | success | 14.4882s |
| 2 | success | 14.4179s |

Implementation detail: verifier is Z3 over a sound interval over-approximation of the sinusoidal term, not dReal over the exact transcendental dynamics.

### ex2 / NCC

Files: `results/fix_ncc/ex2_ncc_ts_vec_seed{0,1,2}.json`.

| seed | result | elapsed | xi | eta_train | eta_cert |
|---:|---|---:|---:|---:|---:|
| 0 | certified | 119.6491s | 0.1 | 0.05 | 0.0470 |
| 1 | certified | 121.6096s | 0.1 | 0.05 | 0.0470 |
| 2 | certified | 119.9672s | 0.1 | 0.05 | 0.0465 |

Implementation detail: training is vectorized and certification is an exhaustive finite-grid check plus the NCC Lipschitz theorem condition, not random sampling.

## Important modeling caveat: guarded transition relation

The current code verifies CC/NCC obligations only for transitions satisfying

```text
x in X and f(x) in X
```

For `ex2`, this materially changes the difficulty.  Under the CC-paper 2D Kuramoto dynamics

```text
f_i(x) = x_i + tau*Omega + 1.69 + K*t_s*sin(x_j-x_i) - 0.532*tau*x_i^2
```

most unsafe states in `Xu=[5*pi/6,8*pi/9]` step outside `X=[0,8*pi/9]^2`.  Therefore the accepting edge can become vacuous under the guarded relation.

Concrete check for current ex2/CC seed0 certificate:

- With the current guarded verifier, the certificate is accepted.
- If the `f(x) in X` guard is removed from C1, Z3 immediately finds a counterexample on the accepting edge:

```text
x=(11/4, 0), xn=(33085/8192, 13853/8192)
```

This explains why the current ex2/CC certificate is extremely simple (`T_1_0=-1`) and why it is much faster than the paper's reported 1h50m CEGIS loop.

Conclusion: the current ex2/CC and ex2/NCC results are formal for the guarded-transition semantics, but they are not an apples-to-apples timing reproduction of the original CC/NCC paper unless the paper baseline is also interpreted with the same guarded transition relation.

## Formula check against CC paper PDF

The CC paper PDF shows the 2D Kuramoto update as

```text
[x1, x2]^T + [tau*Omega + 1.69, tau*Omega + 1.69]^T
  + K*t_s*[sin(x2-x1), sin(x1-x2)]^T
  - 0.532*tau*[x1^2, x2^2]^T.
```

So `+1.69` is not multiplied by `t_s` in the CC paper.  `latest-paper/casestudy.tex` currently writes `+ t_s [Omega+1.69, Omega+1.69]^T`, which appears inconsistent with the CC paper and with the current code.

## Why NCC can be faster than the NCC paper

The current ex2/NCC run should not be described as a direct reproduction of the NCC paper timing:

1. The NCC paper reports 1D Kuramoto (`epsilon=0.02`, `eta=0.01`, `L=0.2798`, about 5h) and a different two-room setup (`epsilon=0.5`, `eta=0.2`, `L=0.6814`, about 4h).  Our ex2/NCC is the 2D Kuramoto benchmark with different grid/margin settings.
2. Current ex2/NCC uses `xi=0.1`, `eta_train=0.05`, and `eta_cert≈0.047`, not the NCC paper's reported hyperparameters.
3. Current ex2/NCC uses vectorized PyTorch batches and chunked exhaustive checks.  The original description is compatible with a much slower implementation that evaluates many grid constraints less efficiently.
4. The same guarded-transition caveat applies: the finite premise is checked only on `x in X and f(x) in X` transition representatives.
5. Formality is still by exhaustive finite-grid checking plus the Lipschitz margin `L*xi/2 - eta_cert <= 0`; it is not justified by simulation or random tests.

## Practical reporting rule

Until we align the transition semantics with the original CC/NCC baseline, report these fast ex2 CC/NCC results as:

> formal under the guarded transition relation used by our implementation,

not as a direct speedup over the original CC/NCC paper's exact experimental setup.

If the target is an apples-to-apples baseline, rerun CC/NCC with the same transition semantics as the original implementation.  In particular, do not discard obligations solely because `f(x)` lies outside the declared box unless this is explicitly part of the system semantics.
