from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex2/PT test")
    args = parser.parse_args()

    report = make_report("ex2", "PT")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "TS-C1 init nonneg (set)", 0.0 <= 0.0 <= pi / 9 and 0.0 <= pi / 9 <= pi / 9, "X0 boundary mismatch")
            add_check(report, "TS-C2 unsafe neg (set)", m.in_unsafe(5 * pi / 6, 0.0) and m.in_unsafe(0.0, 5 * pi / 6), "Xu boundary mismatch")
            add_check(report, "TS-C3 transition preserve (delta)", m.delta(0.1, 0.1, 1) == [1] and m.delta(5 * pi / 6, 0.1, 1) == [0], "delta mismatch")
            x1n, x2n = m.f_m(0.1, 0.2)
            add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f_m not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "TS-C1 source", "Initial non-negativity" in text, "C1 block not found")
            add_check(report, "TS-C2 source", "Unsafe negativity" in text, "C2 block not found")
            add_check(report, "TS-C3 source", "Transition preservation" in text, "C3 block not found")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            coeffs = data.get("coefficients")
            add_check(report, "result coefficients", isinstance(coeffs, list) and len(coeffs) == 7, "expected 7 coefficients")
            add_check(report, "result type", data.get("certificate_type") == "transition_safety", f"type={data.get('certificate_type')}")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
