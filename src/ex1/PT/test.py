from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex1/PT test")
    args = parser.parse_args()

    report = make_report("ex1", "PT")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "TS-C1 init nonneg (set)", (4 * pi / 9) <= (4 * pi / 9) <= (5 * pi / 9), "X0 interval malformed")
            add_check(report, "TS-C2 unsafe neg (set)", m.in_unsafe(7 * pi / 9) and m.in_unsafe(8 * pi / 9), "Xu boundary mismatch")
            add_check(report, "TS-C3 transition preserve (delta)", m.delta(0.0, 1) == [1] and m.delta(7 * pi / 9, 1) == [0], "delta mismatch")
            add_check(report, "dynamics finite", math.isfinite(m.f_m(1.0)), "f_m(1.0) is not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "TS-C1 source", "4 * PI / 9" in text and "5 * PI / 9" in text, "X0 config not found in source")
            add_check(report, "TS-C2 source", "7 * PI / 9" in text and "8 * PI / 9" in text, "Xu config not found in source")
            add_check(report, "TS-C3 source", "return [0] if in_unsafe(x) else [1]" in text, "delta logic not found in source")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            coeffs = data.get("coefficients")
            add_check(report, "result coefficients", isinstance(coeffs, list) and len(coeffs) == 9, "expected 9 coefficients")
            add_check(report, "result success", isinstance(data.get("success"), bool), "missing success bool")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
