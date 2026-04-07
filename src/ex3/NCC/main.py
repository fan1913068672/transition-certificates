import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result


def in_x(x1, x2):
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x0(x1, x2):
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def in_vf(x1, x2):
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def labels(x1, x2):
    return int(in_x0(x1, x2)), int(in_vf(x1, x2))


def f(x1, x2):
    alpha, theta, mu, th, te = 0.004, 0.01, 0.15, 40.0, 0.0

    def u(x):
        return 0.59 - 0.011 * x

    return (
        (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * th * u(x1) + theta * te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * th * u(x2) + theta * te,
    )


def delta_main(q, x1, x2):
    a0, _ = labels(x1, x2)
    if q == 0:
        return [1] if a0 == 1 else [2]
    if q == 1:
        return [1]
    return [2]


def delta_closure(q, x1, x2):
    a0, a1 = labels(x1, x2)
    if q == 0:
        if (a0, a1) == (1, 1):
            return [2]
        if (a0, a1) == (0, 1):
            return [1, 3]
        if (a0, a1) == (1, 0):
            return [1]
        return [3]
    if q == 1:
        return [2] if a1 == 1 else [1]
    if q == 2:
        return [2]
    return [3]


class Net(nn.Module):
    def __init__(self, hidden=80):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))).squeeze(-1)


def make_grid(a, b, step):
    vals = []
    x = a
    while x <= b + 1e-12:
        vals.append(round(x, 8))
        x += step
    return vals


def t_input(x1, x2, q, y1, y2, j):
    return torch.tensor([[x1, x2, float(q), y1, y2, float(j)]], dtype=torch.float32)


def g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2, q0):
    fx1, fx2 = f(x1, x2)
    t_x_fx = net(t_input(x1, x2, q, fx1, fx2, qp))[0]
    t_x_y = net(t_input(x1, x2, q, y1, y2, j))[0]
    t_fx_y = net(t_input(fx1, fx2, qp, y1, y2, j))[0]
    t_x0_y = net(t_input(x01, x02, q0, y1, y2, j))[0]
    t_x0_yp = net(t_input(x01, x02, q0, yp1, yp2, jp))[0]
    t_y_yp = net(t_input(y1, y2, j, yp1, yp2, jp))[0]

    g1 = t_x_fx
    g2 = t_x_y - t_fx_y
    g3 = (1 - lam1) * t_x0_y - t_x0_yp - lam2 * t_y_yp
    return g1, g2, g3


def spectral_norm(mat: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(mat).max().item())


def network_lipschitz(net: Net) -> float:
    with torch.no_grad():
        return spectral_norm(net.fc1.weight.detach()) * spectral_norm(net.fc2.weight.detach())


def estimate_lf_numeric(xmin=20.0, xmax=34.0, step=0.5):
    # numerical Jacobian row-sum bound (infinity norm) for f
    xs = make_grid(xmin, xmax, step)
    h = 1e-4
    best = 0.0
    for x1 in xs:
        for x2 in xs:
            f0 = f(x1, x2)
            fx1 = f(x1 + h, x2)
            fx2 = f(x1, x2 + h)
            d11 = (fx1[0] - f0[0]) / h
            d12 = (fx2[0] - f0[0]) / h
            d21 = (fx1[1] - f0[1]) / h
            d22 = (fx2[1] - f0[1]) / h
            row1 = abs(d11) + abs(d12)
            row2 = abs(d21) + abs(d22)
            best = max(best, row1, row2)
    return best


def train(mode='closure', epochs=2200, lr=1e-3, xi=0.01, eta=0.01, lam1=0.1, lam2=0.1, tol=1e-4):
    xi_eff = xi
    if xi_eff <= 0:
        raise ValueError("xi must be > 0")
    xs = make_grid(20.0, 34.0, xi_eff)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]

    if mode == 'closure':
        q_states = [0, 1, 2, 3]
        q0 = 0
        q_acc = [2]
        delta = delta_closure
    else:
        q_states = [0, 1, 2]
        q0 = 0
        q_acc = [1]
        delta = delta_main

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(19)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        for _ in range(240):
            x1, x2 = rng.choice(pts)
            q = rng.choice(q_states)
            qp = rng.choice(delta(q, x1, x2))
            y1, y2 = rng.choice(pts)
            j = rng.choice(q_states)
            yp1, yp2 = rng.choice(pts)
            jp = rng.choice(q_acc)
            x01, x02 = rng.choice(x0)

            g1, g2, g3 = g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2, q0)
            loss = loss + torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)

        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    max_lprime = 0.0
    for _ in range(1800):
        x1, x2 = rng.choice(pts)
        q = rng.choice(q_states)
        qp = rng.choice(delta(q, x1, x2))
        y1, y2 = rng.choice(pts)
        j = rng.choice(q_states)
        yp1, yp2 = rng.choice(pts)
        jp = rng.choice(q_acc)
        x01, x02 = rng.choice(x0)
        g1, g2, g3 = g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2, q0)
        lprime = float((torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)).item())
        max_lprime = max(max_lprime, lprime)

    lt = network_lipschitz(net)
    lf = estimate_lf_numeric(step=1.0)
    l1 = lt * (1.0 + lf)
    l2 = 2.0 * lt * (1.0 + lf)
    l3 = (2.0 + lam1 + lam2) * lt
    l_all = max(l1, l2, l3)
    theorem_margin = l_all * xi_eff / 2.0 - eta
    certified = (max_lprime <= tol) and (theorem_margin <= 0.0)

    payload = {
        'arch': '6-80-1',
        'mode': mode,
        'xi': xi,
        'xi_effective': xi_eff,
        'eta': eta,
        'lambda1': lam1,
        'lambda2': lam2,
        'L_T': lt,
        'L_f': lf,
        'L1': l1,
        'L2': l2,
        'L3': l3,
        'L': l_all,
        'max_lprime_sampled': max_lprime,
        'theorem_margin': theorem_margin,
        'certified': certified,
        'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()},
    }

    Path('saved_models').mkdir(exist_ok=True)
    Path('saved_models/model.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    Path('res.txt').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('saved to saved_models/model.json')
    print(f'certified={certified}, theorem_margin={theorem_margin:.6f}, max_lprime={max_lprime:.6f}')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='res_ncc_ex3.json')
    p.add_argument('--mode', choices=['main', 'closure'], default='closure')
    p.add_argument('--max-iter', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--epochs', type=int, default=2200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grid-step', '--xi', dest='xi', type=float, default=0.01)
    p.add_argument('--dreal-precision', type=float, default=0.0, help='unused (kept for CLI consistency)')
    p.add_argument('--z3-timeout-ms', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--seed', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--qi', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--qj', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--eta', type=float, default=0.01)
    p.add_argument('--lambda1', type=float, default=0.1)
    p.add_argument('--lambda2', type=float, default=0.1)
    p.add_argument('--tol', type=float, default=1e-4)
    args = p.parse_args()
    print_header("ex3", "NCC", "neural_closure_certificate", {"mode": args.mode, "epochs": args.epochs, "lr": args.lr, "xi": args.xi})
    t0 = time.time()
    payload = train(mode=args.mode, epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex3"
    payload["method"] = "NCC"
    payload["elapsed_sec"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
