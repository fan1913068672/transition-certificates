# CC/NCC Consistency Audit (against `main.pdf` appendix + `cc.pdf`/`ncc.pdf`)

## Scope rule used
- **Automata and benchmark configuration**: follow `main.pdf` appendix comparative setting.
- **CC condition semantics (C1/C2/C3)**: check against `cc.pdf` (Closure Certificates).
- **NCC method semantics**: check against `ncc.pdf` (Neural Closure Certificates).

---

## Progress done in this round

### ex1/CC (`src/ex1/CC/main.py`) fixes applied
1. Fixed Z3 decimal parsing bug in `t2float`:
   - from `':'` to `'?'` suffix handling.
2. Fixed C3 initial-domain declaration:
   - from `[0, pi/9]` to `[4pi/9, 5pi/9]` (matches ex1 `X0` config).
3. Fixed undefined variable in synthesis loop print:
   - from `iteration` to `iter_count`.

### ex2/CC and ex3/CC numeric export fixes
- Fixed model decimal parsing from `replace(':', '')` to `replace('?', '')`.

### ex2/CC solver alignment update
- Reworked `src/ex2/CC/main.py` to follow requested solver split:
  - Z3 for candidate synthesis,
  - dReal for continuous-domain counterexample checks (C1/C2/C3).

### ex3/CC automaton branch fix
- Updated closure-mode transition function so that for `q0` and label `(a0=0,a1=1)` it returns `[1,3]`, matching the nondeterministic transition pattern in the appendix-style automaton.

---

## Remaining gaps (to be implemented next)

### CC
- ex2/CC and ex3/CC currently use sampled constraints only; they do not yet implement full continuous-domain counterexample validation loop equivalent to strict CC verification style.
- ex3/CC `closure` automaton branch should be re-checked line-by-line against the finalized automaton in `main.pdf` appendix.

### NCC
Current NCC scripts are neural training baselines, but not yet full `ncc.pdf`-style certified NCC:
1. (Done) Added Eq.(14)-(18)-style training terms using `g1`, `g2`, `g3` and robust margin `eta`.
2. (Done) Added explicit `lambda1`, `lambda2` terms in `g3`.
3. (Done) Added spectral-norm-based network Lipschitz estimation.
4. (Done) Added theorem-margin check: `L * xi / 2 - eta <= 0`.
5. (Done) Added final certification fields: `certified`, `theorem_margin`, and sampled `max_lprime`.

Note:
- The NCC implementation now follows the paper structure and certification logic, while still using practical sampled training/evaluation loops.

---

## Next coding steps (recommended)
1. Normalize ex3/CC automaton transitions to the final appendix definition.
2. Add solver-based validation phase to ex2/CC and ex3/CC.
3. Rebuild NCC training/verification to match `ncc.pdf` equations and theorem check.
