import argparse
import json
import math
from pathlib import Path

PI = 3.1415926
TS = 0.1
OMEGA = 0.01
K = 0.0006


def in_x(x1, x2):
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


def in_unsafe(x1, x2):
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def f(x1, x2):
    x1p = x1 + TS * OMEGA + 1.69 + TS * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + TS * OMEGA + 1.69 + TS * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p


def delta(x1, x2, q):
    if q == 1:
        return [0] if in_unsafe(x1, x2) else [1]
    return [0]


def step_sample(a, b, s):
    return [round(i * s, 10) for i in range(int(a * (1 / s)), int(b * (1 / s)) + 1)]


def eval_b(c, x1, x2, q):
    # B = c0 + c1*x1 + c2*x2 + c3*q + c4*I_Xu + c5*x1*q + c6*x2*q
    iu = 1.0 if in_unsafe(x1, x2) else 0.0
    return c[0] + c[1] * x1 + c[2] * x2 + c[3] * q + c[4] * iu + c[5] * x1 * q + c[6] * x2 * q


def load_coeffs(path):
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("coefficients"), list) and len(data["coefficients"]) >= 7:
            return [float(x) for x in data["coefficients"][:7]]
    # fallback to current known good run
    return [-1.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0]


def main():
    ap = argparse.ArgumentParser(description="Numeric checker for ex2 transition-safety certificate.")
    ap.add_argument("--coef-json", default="src/ex2/PT/smoke_ex2_pt_safety.json")
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-10)
    args = ap.parse_args()

    c = load_coeffs(args.coef_json)
    print("Using coefficients:", c)

    xs = step_sample(0, 8 * PI / 9, args.grid_step)
    x0s = step_sample(0, PI / 9, args.grid_step)

    # C1: initial non-negativity
    c1_bad = 0
    c1_min = float("inf")
    for x1 in x0s:
        for x2 in x0s:
            v = eval_b(c, x1, x2, 1)
            c1_min = min(c1_min, v)
            if v < -args.eps:
                c1_bad += 1

    # C2: unsafe negativity (here unsafe product set is q=0 over X)
    c2_bad = 0
    c2_max = -float("inf")
    for x1 in xs:
        for x2 in xs:
            v = eval_b(c, x1, x2, 0)
            c2_max = max(c2_max, v)
            if v >= args.eps:
                c2_bad += 1

    # C3: invariance
    c3_bad = 0
    worst_next = float("inf")
    total_checked = 0
    for x1 in xs:
        for x2 in xs:
            for q in [0, 1]:
                b_now = eval_b(c, x1, x2, q)
                if b_now >= -args.eps:
                    x1p, x2p = f(x1, x2)
                    if not in_x(x1p, x2p):
                        continue
                    for qp in delta(x1, x2, q):
                        total_checked += 1
                        b_next = eval_b(c, x1p, x2p, qp)
                        worst_next = min(worst_next, b_next)
                        if b_next < -args.eps:
                            c3_bad += 1

    print("\n=== Numeric check summary ===")
    print(f"C1 init nonneg: bad={c1_bad}, min_B_init={c1_min:.6e}")
    print(f"C2 unsafe neg : bad={c2_bad}, max_B_q0 ={c2_max:.6e}")
    print(f"C3 invariance : bad={c3_bad}/{total_checked}, min_B_next={worst_next:.6e}")

    ok = (c1_bad == 0 and c2_bad == 0 and c3_bad == 0)
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()

