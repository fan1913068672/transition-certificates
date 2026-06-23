from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import check_sampled_initial_points, add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex3/ST test")
    args = parser.parse_args()

    report = make_report("ex3", "ST")
    check_sampled_initial_points(report, None, "ex3", args.x0_samples, args.sim_steps)
    try:
        m = load_local_main(__file__)
        add_check(report, "init region", m.in_x0(21, 21) and m.in_x0(24, 24), "X0 boundary mismatch")
        add_check(report, "VF region", m.in_vf(20, 20) and m.in_vf(26, 26), "VF boundary mismatch")
        x1n, x2n = m.f(22.0, 23.0)
        add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f not finite")
        add_check(report, "label function", m.label_value(22, 22) == 3 and m.label_value(30, 30) == 0, "label mismatch")

        case = m.build_case()
        add_check(report, "accepting state", case.automaton.accepting_states == [1], "accepting state mismatch")
        add_check(report, "state-based GFb edge 0ab1", (0, ("a", "b")) in case.automaton.transitions and case.automaton.transitions[(0, ("a", "b"))] == [1], "missing 0 --{a,b}--> 1")
        add_check(report, "state-based GFb edge 2b1", (2, ("b",)) in case.automaton.transitions and case.automaton.transitions[(2, ("b",))] == [1], "missing 2 --{b}--> 1")
        result = m.analyze_case(case)
        add_check(report, "strict wong flag", result.get("strict_wong_pf3") is True, "strict wong flag missing")
        add_check(report, "inconclusive expected", result.get("status") == "inconclusive", "ex3 should remain inconclusive under strict PF^3")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result meta", data.get("example") == "ex3" and data.get("method") == "ST", "wrong example/method")
            add_check(report, "result type", data.get("certificate_type") == "state_triplet_barrier", "wrong certificate_type")
            add_check(report, "negated spec", data.get("negated_specification") == "a ? GF b", "wrong negated specification")
    except Exception as e:
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
