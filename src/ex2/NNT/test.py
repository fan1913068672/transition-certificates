from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common_test_utils import find_latest_model, load_model
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex2/NNT test")
    parser.add_argument("--check-model", action="store_true", help="evaluate latest model if available")
    args = parser.parse_args()

    report = make_report("ex2", "NNT")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "TS-C1 init nonneg (set)", m.In_X0(0.0, 0.0) and m.In_X0(pi / 9, pi / 9), "X0 boundary mismatch")
            xu_mid = (5 * pi / 6 + 8 * pi / 9) / 2.0
            add_check(report, "TS-C2 unsafe neg (set)", m.In_Unsafe(xu_mid, 0.0) and m.In_Unsafe(0.0, xu_mid), "Xu boundary mismatch")
            add_check(report, "TS-C3 transition preserve (delta)", m.delta(0.1, 0.1, 1) == [1], f"delta safe={m.delta(0.1,0.1,1)}")
            x1n, x2n = m.f_m(0.1, 0.2)
            add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f_m not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "TS-C1 source", "In_X0_Cond" in text, "C1 set not found")
            add_check(report, "TS-C2 source", "In_Unsafe_Cond" in text, "C2 set not found")
            add_check(report, "TS-C3 source", "transition" in text.lower(), "C3 logic not found")

        if args.check_model:
            model_path = find_latest_model(Path(__file__).parent)
            model = load_model(model_path, input_dim=3)
            x1 = random.uniform(0.0, math.pi / 9)
            x2 = random.uniform(0.0, math.pi / 9)
            b = float(model(x1, x2, 1).item())
            add_check(report, "model output finite", math.isfinite(b), f"B={b}")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result success key", "success" in data, "missing success field")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
