from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def check_t2float_suffix(py: str, text: str, issues: list[str]) -> None:
    if "as_decimal" in text and "s[-1] == ':'" in text:
        issues.append(f"{py}: uses old t2float suffix ':' (should be '?').")


def check_unsafe_cond_2d(py: str, text: str, issues: list[str]) -> None:
    # Heuristic: 2D unsafe condition should mention both x1 and x2 in the condition body.
    for fn in ("In_Unsafe_Cond", "in_unsafe_cond"):
        m = re.search(rf"def\s+{fn}\s*\(([^)]*)\)\s*:\s*([\s\S]*?)(?:\n\s*def\s+|\Z)", text)
        if not m:
            continue
        args = [a.strip() for a in m.group(1).split(",") if a.strip()]
        body = m.group(2)
        if len(args) >= 2:
            a1, a2 = args[0], args[1]
            if a1 in body and a2 not in body:
                issues.append(f"{py}: {fn} seems to use '{a1}' only (missing '{a2}' in body).")


def check_interval_order(py: str, text: str, issues: list[str]) -> None:
    # Simple pattern: x >= low and x <= high in one line; warn if same var appears with swapped numbers.
    patt = re.compile(r"([A-Za-z_]\w*)\s*>=\s*([^,\n]+?)\s*[,)]\s*\1\s*<=\s*([^,\n)]+)")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if ">=" in line and "<=" in line:
            # only heuristic check for obvious literal inversions
            mm = re.search(r"(\d+(?:\.\d+)?)\s*>\s*(\d+(?:\.\d+)?)", line)
            if mm:
                issues.append(f"{py}:{line_no}: suspicious literal comparison in interval expression.")


def run() -> int:
    py_files = list(SRC.rglob("*.py"))
    issues: list[str] = []

    for p in py_files:
        t = read(p)
        rel = str(p.relative_to(ROOT))
        check_t2float_suffix(rel, t, issues)
        check_unsafe_cond_2d(rel, t, issues)
        check_interval_order(rel, t, issues)

    print(f"[sanity] scanned {len(py_files)} Python files under {SRC}")
    if issues:
        print("[sanity] issues found:")
        for it in issues:
            print(" -", it)
        return 1
    print("[sanity] no obvious x1/x2 or interval-pattern issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

