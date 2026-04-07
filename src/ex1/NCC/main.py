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

PI = 3.1415926


def in_x(x):
    return 0.0 <= x <= 2 * PI


def in_x0(x):
    return 4 * PI / 9 <= x <= 5 * PI / 9


def in_xu(x):
    return 7 * PI / 9 <= x <= 8 * PI / 9


def f(x):
    ts, omega, k = 0.1, 0.01, 0.0006
    return x + ts * omega + ts * k * math.sin(-x) - 0.532 * x * x + 1.69


def delta(q, x):
    if q == 1:
        return [0] if in_xu(x) else [1]
    return [0]


class Net(nn.Module):
    def __init__(self, hidden=80):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


def make_grid(a, b, step):
    vals = []
    x = a
    while x <= b + 1e-12:
        vals.append(round(x, 8))
        x += step
    return vals


def t_input(x, q, y, j):
    return torch.tensor([[x, float(q), y, float(j)]], dtype=torch.float32)


def g_values(net, x, q, qp, x0, y, j, yp, jp, lam1, lam2):
    fx = f(x)
    t_x_fx = net(t_input(x, q, fx, qp))[0]
    t_x_y = net(t_input(x, q, y, j))[0]
    t_fx_y = net(t_input(fx, qp, y, j))[0]
    t_x0_y = net(t_input(x0, 1, y, j))[0]
    t_x0_yp = net(t_input(x0, 1, yp, jp))[0]
    t_y_yp = net(t_input(y, j, yp, jp))[0]

    g1 = t_x_fx
    g2 = t_x_y - t_fx_y
    g3 = (1 - lam1) * t_x0_y - t_x0_yp - lam2 * t_y_yp
    return g1, g2, g3


def spectral_norm(mat: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(mat).max().item())


def network_lipschitz(net: Net) -> float:
    with torch.no_grad():
        s1 = spectral_norm(net.fc1.weight.detach())
        s2 = spectral_norm(net.fc2.weight.detach())
    return s1 * s2


def estimate_lf_ex1() -> float:
    # conservative bound on |f'(x)| over [0, 2pi]
    # f'(x)=1 - 0.00006*cos(x) - 1.064x
    return 6.0


def build_samples(xi):
    xs = make_grid(0.0, 2 * PI, xi)
    x0s = [x for x in xs if in_x0(x)]
    q_states = [0, 1]
    q_acc = [0]
    return xs, x0s, q_states, q_acc


def train(epochs=1500, lr=1e-3, xi=0.01, eta=0.01, lam1=0.1, lam2=0.1, tol=1e-4):
    xs, x0s, q_states, q_acc = build_samples(xi)
    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)

    rng = random.Random(7)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        for _ in range(200):
            x = rng.choice(xs)
            q = rng.choice(q_states)
            qp = rng.choice(delta(q, x))
            y = rng.choice(xs)
            j = rng.choice(q_states)
            yp = rng.choice(xs)
            jp = rng.choice(q_acc)
            x0 = rng.choice(x0s)

            g1, g2, g3 = g_values(net, x, q, qp, x0, y, j, yp, jp, lam1, lam2)
            # l'_s from ncc paper (Eq.18 style)
            loss = loss + torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)

        loss.backward()
        opt.step()

        if ep % 200 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    # post check on sampled points
    max_lprime = 0.0
    for _ in range(1200):
        x = rng.choice(xs)
        q = rng.choice(q_states)
        qp = rng.choice(delta(q, x))
        y = rng.choice(xs)
        j = rng.choice(q_states)
        yp = rng.choice(xs)
        jp = rng.choice(q_acc)
        x0 = rng.choice(x0s)
        g1, g2, g3 = g_values(net, x, q, qp, x0, y, j, yp, jp, lam1, lam2)
        lprime = float((torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)).item())
        max_lprime = max(max_lprime, lprime)

    lt = network_lipschitz(net)
    lf = estimate_lf_ex1()
    l1 = lt * (1.0 + lf)
    l2 = 2.0 * lt * (1.0 + lf)
    l3 = (2.0 + lam1 + lam2) * lt
    l_all = max(l1, l2, l3)

    theorem_margin = l_all * xi / 2.0 - eta
    certified = (max_lprime <= tol) and (theorem_margin <= 0.0)

    Path('saved_models').mkdir(exist_ok=True)
    payload = {
        'arch': '4-80-1',
        'xi': xi,
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
        'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}
    }
    Path('saved_models/model.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    Path('res.txt').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('saved to saved_models/model.json')
    print(f'certified={certified}, theorem_margin={theorem_margin:.6f}, max_lprime={max_lprime:.6f}')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='res_ncc_ex1.json')
    p.add_argument('--max-iter', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--epochs', type=int, default=1500)
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
    print_header("ex1", "NCC", "neural_closure_certificate", {"epochs": args.epochs, "lr": args.lr, "xi": args.xi})
    t0 = time.time()
    payload = train(epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex1"
    payload["method"] = "NCC"
    payload["elapsed_sec"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
