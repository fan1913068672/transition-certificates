"""Quick consistency check between main.pdf settings and cited CC/NCC papers.

Usage:
    python baseline_config_check.py
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Issue:
    item: str
    main_pdf: str
    cited_paper: str
    level: str = "warning"


def collect_issues() -> List[Issue]:
    issues: List[Issue] = []

    # NCC network size (now aligned)
    issues.append(Issue(
        item="NCC hidden neurons",
        main_pdf="Table ncc_config uses (4/6)-80-1",
        cited_paper="ncc.pdf case studies state one hidden layer with 80 neurons",
        level="info",
    ))

    # Ex3 automaton mismatch in manuscript itself
    issues.append(Issue(
        item="ex3 CC automaton definition",
        main_pdf="casestudy uses negation a ? GF b (accepting self-loop behavior)",
        cited_paper="appendix comparative CC section lists NBA for a0 ? F a1 with Q={0,1,2,3}, Acc={2}",
        level="warning",
    ))

    # two-room physical ranges mismatch between ncc and this manuscript benchmarks
    issues.append(Issue(
        item="Two-room state ranges in NCC source paper",
        main_pdf="X=[20,34]^2, X0=[21,24]^2",
        cited_paper="ncc.pdf two-room case uses X=[9,19]^2, X0=[9,10]^2",
        level="info",
    ))

    # unspecified hyperparameters in main for NCC training
    issues.append(Issue(
        item="NCC training hyperparameters completeness",
        main_pdf="gives architecture + xi/eta + spectral-norm Lipschitz estimation",
        cited_paper="does not provide per-benchmark exact epoch schedule for this repo's settings",
        level="warning",
    ))

    return issues


def main() -> None:
    issues = collect_issues()
    print("=== Configuration consistency check (main.pdf vs cited baselines) ===")
    for i, iss in enumerate(issues, 1):
        print(f"[{i}] {iss.level.upper()} - {iss.item}")
        print(f"    main.pdf:    {iss.main_pdf}")
        print(f"    cited paper: {iss.cited_paper}")


if __name__ == "__main__":
    main()
