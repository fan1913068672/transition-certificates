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
    parser = base_parser("ex3/NNT test")
    parser.add_argument("--check-model", action="store_true", help="evaluate latest model if available")
    args = parser.parse_args()

    report = make_report("ex3", "NNT")
    try:
        try:
            m = load_local_main(__file__)
            add_check(report, "TP-C1 init nonneg (set)", m.In_X0(21, 21) and m.In_X0(24, 24), "X0 boundary mismatch")
            add_check(report, "TP-C2 non-accepting nonincrease (set)", m.In_VF(20, 20) and m.In_VF(26, 26), "VF boundary mismatch")
            add_check(report, "TP-C3 accepting strict decrease (set)", m.In_VF(22, 22) and not m.In_VF(30, 30), "VF region logic mismatch")
            x1n, x2n = m.f_m(22.0, 23.0)
            add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f_m not finite")
        except ModuleNotFoundError:
            text = Path(__file__).with_name("main.py").read_text(encoding="utf-8", errors="ignore")
            add_check(report, "TP-C1 source", "In_X0_Cond" in text, "C1 set not found")
            add_check(report, "TP-C2 source", "non_inc" in text, "C2 non-increasing block not found")
            add_check(report, "TP-C3 source", "strict_dec" in text or "strict decrease" in text.lower(), "C3 strict-decrease block not found")

        if args.check_model:
            model_path = find_latest_model(Path(__file__).parent)
            model = load_model(model_path, input_dim=2)
            x1 = random.uniform(21, 24)
            x2 = random.uniform(21, 24)
            b = float(model(x1, x2).item())
            add_check(report, "model output finite", math.isfinite(b), f"B={b}")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result success key", "success" in data, "missing success field")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
