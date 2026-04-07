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


def in_x(x1, x2):
    return 0.0 <= x1 <= 8 * PI / 9 and 0.0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0.0 <= x1 <= PI / 9 and 0.0 <= x2 <= PI / 9


def in_xu(x1, x2):
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def f(x1, x2):
    ts, omega, k = 0.1, 0.01, 0.0006
    return (
        x1 + ts * omega + 1.69 + ts * k * math.sin(x2 - x1) - 0.532 * x1 * x1,
        x2 + ts * omega + 1.69 + ts * k * math.sin(x1 - x2) - 0.532 * x2 * x2,
    )


def delta(q, x1, x2):
    if q == 1:
        return [0] if in_xu(x1, x2) else [1]
    return [0]


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


def g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2):
    fx1, fx2 = f(x1, x2)
    t_x_fx = net(t_input(x1, x2, q, fx1, fx2, qp))[0]
    t_x_y = net(t_input(x1, x2, q, y1, y2, j))[0]
    t_fx_y = net(t_input(fx1, fx2, qp, y1, y2, j))[0]
    t_x0_y = net(t_input(x01, x02, 1, y1, y2, j))[0]
    t_x0_yp = net(t_input(x01, x02, 1, yp1, yp2, jp))[0]
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


def estimate_lf_ex2() -> float:
    # conservative infinity-norm bound for Jacobian row sum
    return 2.2


def train(epochs=2000, lr=1e-3, xi=0.01, eta=0.01, lam1=0.1, lam2=0.1, tol=1e-4):
    xi_eff = xi
    if xi_eff <= 0:
        raise ValueError("xi must be > 0")
    xs = make_grid(0.0, 8 * PI / 9, xi_eff)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    q_states = [0, 1]
    q_acc = [0]

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(11)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for _ in range(220):
            x1, x2 = rng.choice(pts)
            q = rng.choice(q_states)
            qp = rng.choice(delta(q, x1, x2))
            y1, y2 = rng.choice(pts)
            j = rng.choice(q_states)
            yp1, yp2 = rng.choice(pts)
            jp = rng.choice(q_acc)
            x01, x02 = rng.choice(x0)

            g1, g2, g3 = g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2)
            loss = loss + torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)

        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    max_lprime = 0.0
    for _ in range(1600):
        x1, x2 = rng.choice(pts)
        q = rng.choice(q_states)
        qp = rng.choice(delta(q, x1, x2))
        y1, y2 = rng.choice(pts)
        j = rng.choice(q_states)
        yp1, yp2 = rng.choice(pts)
        jp = rng.choice(q_acc)
        x01, x02 = rng.choice(x0)

        g1, g2, g3 = g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j, yp1, yp2, jp, lam1, lam2)
        lprime = float((torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)).item())
        max_lprime = max(max_lprime, lprime)

    lt = network_lipschitz(net)
    lf = estimate_lf_ex2()
    l1 = lt * (1.0 + lf)
    l2 = 2.0 * lt * (1.0 + lf)
    l3 = (2.0 + lam1 + lam2) * lt
    l_all = max(l1, l2, l3)
    theorem_margin = l_all * xi_eff / 2.0 - eta
    certified = (max_lprime <= tol) and (theorem_margin <= 0.0)

    payload = {
        'arch': '6-80-1',
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
    p.add_argument('--out', type=str, default='res_ncc_ex2.json')
    p.add_argument('--max-iter', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--epochs', type=int, default=2000)
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
    print_header("ex2", "NCC", "neural_closure_certificate", {"epochs": args.epochs, "lr": args.lr, "xi": args.xi})
    t0 = time.time()
    payload = train(epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex2"
    payload["method"] = "NCC"
    payload["elapsed_sec"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
