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
    parser = base_parser("ex1/NNT test")
    parser.add_argument("--check-model", action="store_true", help="evaluate latest model if available")
    args = parser.parse_args()

    report = make_report("ex1", "NNT")
    try:
        try:
            m = load_local_main(__file__)
            pi = m.PI
            add_check(report, "TS-C1 init nonneg (set)", m.in_initial_set(4 * pi / 9) and m.in_initial_set(5 * pi / 9), "X0 boundary mismatch")
            add_check(report, "TS-C2 unsafe neg (set)", m.in_unsafe_set_numeric(7 * pi / 9) and m.in_unsafe_set_numeric(8 * pi / 9), "Xu boundary mismatch")
            add_check(report, "TS-C3 transition preserve (delta)", m.compute_mode_transition(0.0, 1) == [1], f"delta at safe point: {m.compute_mode_transition(0.0, 1)}")
            add_check(report, "dynamics finite", math.isfinite(m.system_dynamics_numeric(1.0)), "f(1.0) is not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "TS-C1 source", "4*PI/9" in text and "PI*5 / 9" in text, "initial-set config not found")
            add_check(report, "TS-C2 source", "PI / 9 * 7" in text and "PI / 9 * 8" in text, "unsafe-set config not found")
            add_check(report, "TS-C3 transition preserve (source)", "compute_mode_transition" in text and "return [0]" in text, "transition logic not found")

        if args.check_model:
            model_path = find_latest_model(Path(__file__).parent)
            model = load_model(model_path, input_dim=2)
            x = random.uniform(4 * math.pi / 9, 5 * math.pi / 9)
            b = float(model(x, 1).item())
            add_check(report, "model output finite", math.isfinite(b), f"B={b}")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result success key", "success" in data, "missing success field")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
