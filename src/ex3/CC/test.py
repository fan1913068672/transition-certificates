from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex3/CC test")
    args = parser.parse_args()

    report = make_report("ex3", "CC")
    try:
        m = load_local_main(__file__)
        add_check(report, "CC-C1 closure seed", isinstance(m.delta_closure_paper(0, 22, 22), list), "delta_closure output is not list")
        add_check(report, "CC-C2 closure propagation", m.delta_main_pt(0, 22, 22) == [1], f"delta_main_pt={m.delta_main_pt(0,22,22)}")
        add_check(report, "CC-C3 accepting decrease domain", m.in_x0(21, 21) and m.in_x0(24, 24), "X0 boundary mismatch")
        add_check(report, "vf region config", m.in_vf(20, 20) and m.in_vf(26, 26), "VF boundary mismatch")
        x1n, x2n = m.f(22.0, 23.0)
        add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f not finite")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result mode", data.get("mode") in {"main", "closure"}, f"mode={data.get('mode')}")
            t00 = data.get("T_0_0")
            add_check(report, "result coeff length", isinstance(t00, list) and len(t00) == 11, "expected 11 coefficients in T_0_0")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
