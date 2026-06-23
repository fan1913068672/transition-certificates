from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from test_report_utils import check_sampled_initial_points, add_check, add_exception, base_parser, load_local_main, make_report, save_and_print


def main() -> int:
    parser = base_parser("ex1/NCC test")
    parser.add_argument("--train-smoke", action="store_true", help="run a 1-epoch smoke training")
    args = parser.parse_args()

    report = make_report("ex1", "NCC")
    check_sampled_initial_points(report, None, "ex1", args.x0_samples, args.sim_steps)
    try:
        m = load_local_main(__file__)
        pi = m.PI
        add_check(report, "NCC domain", m.in_x(0.0) and m.in_x(2 * pi), "domain mismatch")
        add_check(report, "NCC-g1 domain/transition", m.delta(0, 1.0) == [0] and m.delta(1, 1.0) == [1], "delta mismatch")
        add_check(report, "NCC-g2 closure set", m.in_xu(7 * pi / 9) and m.in_xu(8 * pi / 9), "Xu boundary mismatch")
        add_check(report, "NCC-g3 initial set", m.in_x0(4 * pi / 9) and m.in_x0(5 * pi / 9), "X0 boundary mismatch")
        add_check(report, "dynamics finite", math.isfinite(m.f(1.0)), "f(1.0) is not finite")
        xs, x0s, q_states, q_acc = m.build_samples(0.1)
        add_check(report, "automaton state set", q_states == [0, 1], f"q_states={q_states}")
        add_check(report, "accepting state set", q_acc == [0], f"q_acc={q_acc}")
        add_check(report, "label partition", m.label_of(4 * pi / 9) == "n" and m.label_of(7 * pi / 9) == "a", "label_of mismatch")

        if args.train_smoke:
            proc = subprocess.run(
                [sys.executable, "main.py", "--epochs", "1", "--xi", "0.01", "--eta", "0.01"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=False,
            )
            add_check(report, "smoke train exit", proc.returncode == 0, proc.stderr[-300:] if proc.returncode else "")

        if args.result:
            data = json.loads(Path(args.result).read_text(encoding="utf-8"))
            add_check(report, "result meta", data.get("example") == "ex1" and data.get("method") == "NCC", "wrong example/method")
            add_check(report, "result type", data.get("certificate_type") == "neural_closure_certificate", "wrong certificate_type")
            add_check(report, "result arch", data.get("arch") == "4-80-1", f"arch={data.get('arch')}")
            add_check(report, "result lipschitz", isinstance(data.get("L_T"), (float, int)), "missing L_T")
            add_check(report, "result backend", isinstance(data.get("backend"), dict), "missing backend")
            add_check(report, "result automaton", data.get("automaton", {}).get("accepting_states") == [0], "wrong accepting_states")
    except Exception as e:
        add_exception(report, e)

    return save_and_print(report, args.out, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
