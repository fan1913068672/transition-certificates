# Transition Certificates Artifact

This directory contains the current reproducibility artifact for the paper.

## Layout

- `src/` contains the implementation and example-specific scripts.
- `results/` contains small paper-relevant summaries and result files retained in Git.

## Entry Points

- Example implementations: `src/ex1/`, `src/ex2/`, `src/ex3/`
- Shared utilities: `src/closure_smt.py`, `src/state_triplet_smt.py`, `src/run_output_utils.py`

Large raw result pools and remote run logs are tracked in the parent workflow repository as manifests, not copied here wholesale.

## Review-repair evidence boundary

The R118 `cc_ex3` run is a bounded Z3/template-capacity diagnostic. It may be
used to document solver search behavior, status labels, small summaries, and
checksum-backed reproducibility fields. It is not a counterexample to the
transition-certificate soundness theorem, not evidence that an external paper
is wrong, and not a certificate-success artifact unless a later manifest records
an explicit successful certificate row.
