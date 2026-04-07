from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex3/PT test")
    parser.add_argument("--qi", type=int, default=1)
    parser.add_argument("--qj", type=int, default=1)
    args = parser.parse_args()

    report = make_report("ex3", "PT")
    try:
        m = load_local_main(__file__)
        add_check(report, "TP-C1 init nonneg (set)", m.in_x0(21, 21) and m.in_x0(24, 24), "X0 boundary mismatch")
        add_check(report, "TP-C2 non-accepting nonincrease (set)", m.in_vf(20, 20) and m.in_vf(26, 26), "VF boundary mismatch")
        add_check(report, "TP-C3 accepting strict decrease (guard)", m.check_trans(22, 22, args.qi, args.qj), "check_trans returned False at representative point")
        x1n, x2n = m.f(22.0, 23.0)
        add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f not finite")
        add_check(report, "in domain", m.in_x(x1n, x2n), "next state out of X at representative point")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result success", isinstance(data.get("success"), bool), "missing success bool")
            coeffs = data.get("coefficients")
            add_check(report, "result coeff length", isinstance(coeffs, list) and len(coeffs) == 10, "expected 10 coefficients")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
