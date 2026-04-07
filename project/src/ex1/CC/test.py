from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex1/CC test")
    args = parser.parse_args()

    report = make_report("ex1", "CC")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "CC-C1 closure seed", m.Automaton.delta(1, False) == [1], f"delta(1,False)={m.Automaton.delta(1, False)}")
            add_check(report, "CC-C2 closure propagation", m.Automaton.delta(1, True) == [0], f"delta(1,True)={m.Automaton.delta(1, True)}")
            add_check(report, "CC-C3 accepting decrease domain", m.In_X0(4 * pi / 9) and m.In_X0(5 * pi / 9), "X0 endpoints not included")
            add_check(report, "unsafe region config", m.In_Unsafe(7 * pi / 9) and m.In_Unsafe(8 * pi / 9), "unsafe endpoints not included")
            add_check(report, "dynamics finite", math.isfinite(m.f_m(1.0)), "f_m(1.0) is not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "CC-C1 source", "check_condition1" in text and "Automaton.delta" in text, "C1 logic not found in source")
            add_check(report, "CC-C2 source", "condition2" in text or "check_condition2" in text, "C2 logic not found in source")
            add_check(report, "CC-C3 source", "condition3" in text or "check_condition3" in text, "C3 logic not found in source")

        if args.result:
            rp = Path(args.result)
            data = json.loads(rp.read_text(encoding="utf-8"))
            coeffs = data.get("coefficients")
            add_check(report, "result coefficients", isinstance(coeffs, list) and len(coeffs) == 20, "expected 20 coefficients")
            add_check(report, "result epsilon", isinstance(data.get("epsilon"), (float, int)), "missing numeric epsilon")
    except Exception as e:  # pragma: no cover - defensive
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
