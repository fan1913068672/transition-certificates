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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def transition_enabled(x1, x2):
    # Domain-restricted transition semantics: only transitions with source
    # and successor in X are part of the verification problem.
    fx1, fx2 = f(x1, x2)
    return in_x(x1, x2) and in_x(fx1, fx2)


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
            return [1]
        if (a0, a1) == (1, 0):
            return [2]
        return [3]
    if q == 1:
        return [1] if a1 == 1 else [2]
    if q == 2:
        return [1] if a1 == 1 else [2]
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


def segment_centers(a, b, max_diam):
    if b <= a:
        return []
    n = max(1, math.ceil((b - a) / max_diam))
    w = (b - a) / n
    return [a + (k + 0.5) * w for k in range(n)]


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


CONTINUOUS_COLS = [0, 1, 3, 4]


def network_lipschitz_tensor(net: Net):
    # q and j are discrete automaton states and are exhaustively enumerated.
    # Only continuous x/y coordinates contribute to the finite-cell radius.
    return torch.linalg.svdvals(net.fc1.weight[:, CONTINUOUS_COLS]).max() * torch.linalg.svdvals(net.fc2.weight).max()


def network_lipschitz(net: Net) -> float:
    with torch.no_grad():
        return spectral_norm(net.fc1.weight.detach()[:, CONTINUOUS_COLS]) * spectral_norm(net.fc2.weight.detach())


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


def train(mode='closure', epochs=2200, lr=1e-3, xi=0.01, eta=0.01, lam1=0.1, lam2=0.1, tol=1e-4, seed=0, lip_reg=1e-2):
    xi_eff = xi
    xs = []
    # Split at AP boundaries for a=X0 and b=XVF so labels are cell-constant.
    cuts = [20.0, 21.0, 24.0, 26.0, 34.0]
    for a, b in zip(cuts[:-1], cuts[1:]):
        xs.extend(segment_centers(a, b, xi_eff))
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    trans_pts = [(x1, x2) for x1, x2 in pts if transition_enabled(x1, x2)]
    x0_axis = segment_centers(21.0, 24.0, xi_eff)
    x0 = [(x1, x2) for x1 in x0_axis for x2 in x0_axis if in_x0(x1, x2)]

    if mode == 'closure':
        q_states = [0, 1, 2, 3]
        q0 = 0
        q_acc = [1]
        delta = delta_closure
    else:
        q_states = [0, 1, 2]
        q0 = 0
        q_acc = [1]
        delta = delta_main

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        for _ in range(240):
            x1, x2 = rng.choice(trans_pts)
            q = rng.choice(q_states)
            qp = rng.choice(delta(q, x1, x2))
            y1, y2 = rng.choice(pts)
            # Eq. (12)/(15): q_l ranges over all automaton states for g2.
            j2 = rng.choice(q_states)
            yp1, yp2 = rng.choice(pts)
            x01, x02 = rng.choice(x0)
            # Eq. (13)/(16): j,j' range over accepting states for g3.
            j3 = rng.choice(q_acc)
            jp = rng.choice(q_acc)

            g1, g2, _ = g_values(net, x1, x2, q, qp, x01, x02, y1, y2, j2, yp1, yp2, jp, lam1, lam2, q0)
            t_x0_y = net(t_input(x01, x02, q0, y1, y2, j3))[0]
            t_x0_yp = net(t_input(x01, x02, q0, yp1, yp2, jp))[0]
            t_y_yp = net(t_input(y1, y2, j3, yp1, yp2, jp))[0]
            g3 = (1 - lam1) * t_x0_y - t_x0_yp - lam2 * t_y_yp
            loss = loss + torch.relu(-g1 + eta) + torch.relu(-g2 + eta) + torch.relu(-g3 + eta)

        loss = loss + lip_reg * network_lipschitz_tensor(net)
        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    max_lprime = 0.0
    min_g1 = float("inf")
    min_g2 = float("inf")
    min_g3 = float("inf")
    with torch.no_grad():
        for x1, x2 in trans_pts:
            for q in q_states:
                for qp in delta(q, x1, x2):
                    fx1, fx2 = f(x1, x2)
                    g1 = net(t_input(x1, x2, q, fx1, fx2, qp))[0]
                    min_g1 = min(min_g1, float(g1.item()))
                    max_lprime = max(max_lprime, float(torch.relu(-g1 + eta).item()))

                    for y1, y2 in pts:
                        for j in q_states:
                            t_x_y = net(t_input(x1, x2, q, y1, y2, j))[0]
                            t_fx_y = net(t_input(fx1, fx2, qp, y1, y2, j))[0]
                            g2 = t_x_y - t_fx_y
                            min_g2 = min(min_g2, float(g2.item()))
                            max_lprime = max(max_lprime, float(torch.relu(-g2 + eta).item()))

        for x01, x02 in x0:
            for y1, y2 in pts:
                for yp1, yp2 in pts:
                    for j in q_acc:
                        for jp in q_acc:
                            t_x0_y = net(t_input(x01, x02, q0, y1, y2, j))[0]
                            t_x0_yp = net(t_input(x01, x02, q0, yp1, yp2, jp))[0]
                            t_y_yp = net(t_input(y1, y2, j, yp1, yp2, jp))[0]
                            g3 = (1 - lam1) * t_x0_y - t_x0_yp - lam2 * t_y_yp
                            min_g3 = min(min_g3, float(g3.item()))
                            max_lprime = max(max_lprime, float(torch.relu(-g3 + eta).item()))

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
        'transition_guard': 'domain_restricted: x in X and f(x) in X',
        'eta': eta,
        'lambda1': lam1,
        'lambda2': lam2,
        'lip_reg': lip_reg,
        'L_T': lt,
        'L_f': lf,
        'L1': l1,
        'L2': l2,
        'L3': l3,
        'L': l_all,
        'max_lprime_sampled': max_lprime,
        'max_lprime_grid': max_lprime,
        'num_grid_points': len(pts),
        'num_transition_grid_points': len(trans_pts),
        'min_g1_grid': min_g1,
        'min_g2_grid': min_g2,
        'min_g3_grid': min_g3,
        'certification_sample_check': 'exhaustive_grid',
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
    p.add_argument('--grid-step', '--xi', dest='xi', type=float, default=0.5)
    p.add_argument('--dreal-precision', type=float, default=0.0, help='unused (kept for CLI consistency)')
    p.add_argument('--z3-timeout-ms', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--seed', type=int, default=0, help='random seed for NCC training')
    p.add_argument('--qi', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--qj', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--eta', type=float, default=0.2)
    p.add_argument('--lambda1', type=float, default=0.1)
    p.add_argument('--lambda2', type=float, default=0.1)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--lip-reg', type=float, default=1e-2, help='Lipschitz regularization weight for NCC training')
    args = p.parse_args()
    set_seed(args.seed)
    print_header("ex3", "NCC", "neural_closure_certificate", {"mode": args.mode, "epochs": args.epochs, "lr": args.lr, "xi": args.xi, "seed": args.seed, "lip_reg": args.lip_reg})
    t0 = time.time()
    payload = train(mode=args.mode, epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol, seed=args.seed, lip_reg=args.lip_reg)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex3"
    payload["method"] = "NCC"
    payload["certificate_type"] = "neural_closure_certificate"
    payload["backend"] = {"train": "pytorch", "certify": "lipschitz_bound"}
    payload["automaton"] = {"states": [0, 1, 2, 3] if args.mode == "closure" else [0, 1, 2], "initial_state": 0, "accepting_states": [1]}
    payload["seed"] = args.seed
    payload["elapsed_sec"] = elapsed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
