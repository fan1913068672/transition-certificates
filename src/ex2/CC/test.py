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
            has_eps = isinstance(data.get("eps"), (float, int))
            has_all_t = all(isinstance(data.get(f"T_{i}_{j}"), list) and len(data[f"T_{i}_{j}"]) == 7 for i in (0, 1) for j in (0, 1))
            add_check(report, "result epsilon", has_eps, "missing eps")
            add_check(report, "result template coeffs", has_all_t, "expected 4 templates with 7 coeffs each")
            if has_eps and has_all_t:
                coeffs = {(i, j): [float(v) for v in data[f"T_{i}_{j}"]] for i in (0, 1) for j in (0, 1)}
                ce_list = m.find_counterexamples(coeffs, float(data["eps"]), precision=1e-4, max_per_kind=1)
                add_check(report, "closure conditions (dReal)", len(ce_list) == 0, f"counterexample={ce_list[:1]}")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
