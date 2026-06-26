# Project Index

## Paper

- Current manuscript root: `paper/`
- Main manuscript: `paper/main.tex`
- Bibliography: `paper/reference.bib`
- Cover letter: `paper/coverletter/letter.tex`

## Artifact

- Current implementation: `artifact/`
- Sanity check: `artifact/tools/check_sanity.py`
- Main experiment families: `artifact/src/ex1/`, `artifact/src/ex2/`, `artifact/src/ex3/`
- Requirements: `artifact/requirements.txt`

## Removed From Git

The repository previously tracked old manuscript variants, generated PDFs/logs, zip/rar backups, downloaded literature PDFs, duplicate code roots, and scratch results. These were removed to keep the repository at the active paper plus reproducibility artifact boundary.

## What Not To Commit

- LaTeX build files.
- Downloaded literature PDFs.
- Old archive snapshots.
- Local model checkpoints and raw result pools.
- Duplicate manuscript roots or duplicate code roots.
