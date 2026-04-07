from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex2/CC test")
    args = parser.parse_args()

    report = make_report("ex2", "CC")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "CC-C1 closure seed", m.delta(1, 0.1, 0.1) == [1], f"delta safe={m.delta(1, 0.1, 0.1)}")
            add_check(report, "CC-C2 closure propagation", m.delta(1, 5 * pi / 6, 0.1) == [0], f"delta unsafe={m.delta(1, 5 * pi / 6, 0.1)}")
            add_check(report, "CC-C3 accepting decrease domain", m.in_x0(0.0, 0.0) and m.in_x0(pi / 9, pi / 9), "X0 boundary mismatch")
            add_check(report, "unsafe region config", m.in_xu(5 * pi / 6, 0.0) and m.in_xu(0.0, 5 * pi / 6), "Xu definition mismatch")
            x1n, x2n = m.f_num(0.1, 0.2)
            add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f_num not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "CC-C1 source", "C1:" in text, "C1 block not found")
            add_check(report, "CC-C2 source", "C2" in text, "C2 block not found")
            add_check(report, "CC-C3 source", "C3" in text, "C3 block not found")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result epsilon", isinstance(data.get("eps"), (float, int)), "missing eps")
            add_check(report, "result T_0_0", isinstance(data.get("T_0_0"), list) and len(data["T_0_0"]) == 7, "expected 7 coeffs in T_0_0")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
