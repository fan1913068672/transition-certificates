from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex3/NCC test")
    parser.add_argument("--mode", choices=["main", "closure"], default="main")
    parser.add_argument("--train-smoke", action="store_true", help="run a 1-epoch smoke training")
    args = parser.parse_args()

    report = make_report("ex3", "NCC")
    try:
        m = load_local_main(__file__)
        add_check(report, "NCC-g3 initial set", m.in_x0(21, 21) and m.in_x0(24, 24), "X0 boundary mismatch")
        add_check(report, "NCC-g2 closure set", m.in_vf(20, 20) and m.in_vf(26, 26), "VF boundary mismatch")
        if args.mode == "main":
            add_check(report, "NCC-g1 domain/transition", m.delta_main(0, 22, 22) == [1], f"delta_main={m.delta_main(0,22,22)}")
        else:
            add_check(report, "NCC-g1 domain/transition", isinstance(m.delta_closure(0, 22, 22), list), "delta_closure output is not list")
        x1n, x2n = m.f(22.0, 23.0)
        add_check(report, "dynamics finite", math.isfinite(x1n) and math.isfinite(x2n), "f not finite")

        if args.train_smoke:
            proc = subprocess.run(
                [sys.executable, "main.py", "--mode", args.mode, "--epochs", "1", "--xi", "0.01", "--eta", "0.01"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=False,
            )
            add_check(report, "smoke train exit", proc.returncode == 0, proc.stderr[-300:] if proc.returncode else "")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result arch", data.get("arch") == "6-80-1", f"arch={data.get('arch')}")
            add_check(report, "result mode", data.get("mode") in {"main", "closure"}, f"mode={data.get('mode')}")
    except Exception as e:  # pragma: no cover
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
