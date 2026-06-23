from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import check_sampled_initial_points, add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex1/CC test")
    args = parser.parse_args()

    report = make_report("ex1", "CC")
    check_sampled_initial_points(report, None, "ex1", args.x0_samples, args.sim_steps)
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "unsafe region config", m.in_unsafe(7 * pi / 9) and m.in_unsafe(8 * pi / 9), "unsafe endpoints not included")
            add_check(report, "initial region config", m.in_x0(4 * pi / 9) and m.in_x0(5 * pi / 9), "X0 endpoints not included")
            add_check(report, "dynamics finite", math.isfinite(m.f_num(1.0)[0]), "f_num(1.0) is not finite")

            case = m.build_case()
            add_check(report, "accepting state", case.accepting_states == [0], f"accepting_states={case.accepting_states}")
            add_check(report, "verify backend", case.verify_backend == "dreal", f"verify_backend={case.verify_backend}")
            add_check(report, "safe edge", case.automaton.transitions[(1, ())] == [1], "missing 1 --{}--> 1")
            add_check(report, "unsafe edge", case.automaton.transitions[(1, ("a",))] == [0], "missing 1 --{a}--> 0")
            add_check(report, "accepting self-loop", case.automaton.transitions[(0, ())] == [0], "missing accepting self-loop")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "fallback verify backend", 'solver_verify": "dreal"' in text or "solver_verify': 'dreal'" in text, "strict dreal verify not found in source")
            add_check(report, "fallback automaton labels", 'label("a")' in text, "a automaton label not found in source")
            add_check(report, "fallback accepting state", 'accepting_states=[0]' in text, "accepting state 0 not found in source")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            coeffs = data.get("coefficients", {})
            add_check(report, "result meta", data.get("example") == "ex1" and data.get("method") == "CC", "wrong example/method")
            add_check(report, "result type", data.get("certificate_type") == "closure_certificate", "wrong certificate_type")
            add_check(report, "result backend", isinstance(data.get("backend"), dict) and data["backend"].get("verify") == "dreal", f"backend={data.get('backend')}")
            add_check(report, "result automaton", data.get("automaton", {}).get("accepting_states") == [0], "wrong accepting_states in result")
            t00 = coeffs.get("T_0_0")
            add_check(report, "result coefficients", isinstance(t00, list) and len(t00) == 3, "expected 3 coefficients in T_0_0")
            if data.get("success"):
                add_check(report, "result epsilon", isinstance(data.get("epsilon"), (float, int)), "missing numeric epsilon")
    except Exception as e:
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
