from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import check_sampled_initial_points, add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex2/ST test")
    args = parser.parse_args()

    report = make_report("ex2", "ST")
    check_sampled_initial_points(report, None, "ex2", args.x0_samples, args.sim_steps)
    try:
        m = load_local_main(__file__)
        pi = m.PI
        add_check(report, "unsafe boundary", m.in_unsafe(5 * pi / 6, 0.0) and m.in_unsafe(0.0, 5 * pi / 6), "unsafe corridor mismatch")
        add_check(report, "delta structure", m.delta(0.1, 0.1, 1) == [1] and m.delta(5 * pi / 6, 0.1, 1) == [0], "delta mismatch")
        x1n, x2n = m.f_m(0.1, 0.2)
        add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f_m not finite")

        case = m.build_case()
        add_check(report, "accepting state", case.automaton.accepting_states == [2], "accepting state mismatch")
        add_check(report, "initial edge", case.automaton.transitions[(0, ("i",))] == [1], "missing 0 --{i}--> 1")
        add_check(report, "unsafe edge", case.automaton.transitions[(1, ("u",))] == [2], "missing 1 --{u}--> 2")
        result_path = Path(args.result) if args.result else Path(__file__).resolve().parent / "res_st_ex2_p1e-3.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            add_check(report, "result type", data.get("certificate_type") == "state_triplet_barrier", "wrong certificate_type")
            add_check(report, "result status", data.get("status") == "proved" and data.get("success") is True, "ex2 ST should be proved")
            ppath = data["accepting_reports"][0]["ppath"]
            add_check(report, "blocked path 0-1-2", ppath[0].get("path") == [0, 1, 2] and ppath[0].get("resolved") is True, "0->1->2 not resolved")
            backend = ppath[0]["triplets"][0]["backend"]
            add_check(report, "formal backend", backend.get("verified") == "z3+dreal", "certificate is not formally verified")
            add_check(
                report,
                "coefficients",
                backend.get("coefficients") == [0.0, -1.0, -1.0, -3.125, 2.0, 2.0],
                "unexpected ex2 ST coefficients",
            )
            edge = ppath[1].get("edge", {})
            add_check(report, "direct edge empty", edge.get("edge") == [0, 2] and edge.get("mode") == "edge-empty" and edge.get("resolved") is True, "0->2 empty edge not resolved")
        else:
            add_check(report, "result file present", False, f"missing {result_path}")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
