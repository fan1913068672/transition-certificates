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


def transition_enabled(x):
    # Domain-restricted transition semantics: only transitions with source
    # and successor in X are part of the verification problem.
    return in_x(x) and in_x(f(x))


def find_transition_upper():
    """Largest x in X with f(x) >= 0.

    The SMT/PT/CC implementations verify transitions under the guard
    x in X and f(x) in X.  The NCC finite-cell theorem requires all guards
    used by the loss to be constant on a cell, so we split the 1-D grid at
    the preimage boundary f(x)=0.
    """

    lo, hi = 0.0, 2 * PI
    # f(lo)>0 and f(hi)<0 for this benchmark.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def delta(q, x):
    lab = label_of(x)
    if q == 1:
        if lab == "a":
            return [0]
        return [1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid automaton state: {q}")


def label_of(x):
    if in_xu(x):
        return "a"
    return "n"


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


def segment_centers(a, b, max_diam):
    if b <= a:
        return []
    n = max(1, math.ceil((b - a) / max_diam))
    w = (b - a) / n
    return [a + (k + 0.5) * w for k in range(n)]


def t_input(x, q, y, j):
    return torch.tensor([[x, float(q), y, float(j)]], dtype=torch.float32, device=DEVICE)


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


CONTINUOUS_COLS = [0, 2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def network_lipschitz_tensor(net: Net):
    # q and j are discrete automaton states and are exhaustively enumerated in
    # the finite premise.  The NCC cell-radius theorem only needs the
    # Lipschitz constant with respect to continuous state coordinates.
    return torch.linalg.svdvals(net.fc1.weight[:, CONTINUOUS_COLS]).max() * torch.linalg.svdvals(net.fc2.weight).max()


def network_lipschitz(net: Net) -> float:
    with torch.no_grad():
        s1 = spectral_norm(net.fc1.weight.detach()[:, CONTINUOUS_COLS])
        s2 = spectral_norm(net.fc2.weight.detach())
    return s1 * s2


def estimate_lf_ex1() -> float:
    # Conservative bound on |f'(x)| over the full paper domain X=[0,2pi].
    # f'(x)=1 - 0.00006*cos(x) - 1.064*x.
    hi = 2 * PI
    candidates = [0.0, hi]
    # f''(x)=0.00006*sin(x)-1.064 < 0 on this interval, so |f'| is maximized
    # at an endpoint; keep a small numerical buffer.
    return max(abs(1.0 - 0.00006 * math.cos(x) - 1.064 * x) for x in candidates) + 1e-6


def build_samples(xi):
    # Partition at AP boundaries and the transition-domain boundary so each
    # one-dimensional cell has constant AP label and transition-enabled guard.
    cuts = sorted(set([0.0, find_transition_upper(), 7 * PI / 9, 8 * PI / 9, 2 * PI]))
    xs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        xs.extend(segment_centers(a, b, xi))
    trans_xs = [x for x in xs if transition_enabled(x)]
    x0s = segment_centers(4 * PI / 9, 5 * PI / 9, xi)
    q_states = [0, 1]
    q_acc = [0]
    return xs, trans_xs, x0s, q_states, q_acc


def train(epochs=1500, lr=1e-3, xi=0.01, eta=0.01, lam1=0.1, lam2=0.1, tol=1e-4, seed=0, lip_reg=1e-2):
    xs, trans_xs, x0s, q_states, q_acc = build_samples(xi)
    net = Net(hidden=80).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=lr)

    rng = random.Random(seed)

    batch_size = 1024
    for ep in range(epochs):
        opt.zero_grad()
        batch = []
        for _ in range(batch_size):
            x = rng.choice(trans_xs)
            q = rng.choice(q_states)
            qp = rng.choice(delta(q, x))
            y = rng.choice(xs)
            # Eq. (12)/(15) quantifies q_l over all Q for g2.
            j2 = rng.choice(q_states)
            yp = rng.choice(xs)
            x0 = rng.choice(x0s)
            # Eq. (13)/(16) quantifies j,j' over Q_acc for g3.
            j3 = rng.choice(q_acc)
            jp = rng.choice(q_acc)
            batch.append((x, float(q), float(qp), f(x), y, float(j2), yp, x0, float(j3), float(jp)))

        inp_x_fx = torch.tensor([[r[0], r[1], r[3], r[2]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x_y = torch.tensor([[r[0], r[1], r[4], r[5]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_fx_y = torch.tensor([[r[3], r[2], r[4], r[5]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x0_y = torch.tensor([[r[7], 1.0, r[4], r[8]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x0_yp = torch.tensor([[r[7], 1.0, r[6], r[9]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_y_yp = torch.tensor([[r[4], r[8], r[6], r[9]] for r in batch], dtype=torch.float32, device=DEVICE)

        g1 = net(inp_x_fx)
        g2 = net(inp_x_y) - net(inp_fx_y)
        g3 = (1 - lam1) * net(inp_x0_y) - net(inp_x0_yp) - lam2 * net(inp_y_yp)
        # l'_s from ncc paper (Eq.18 style).  Use a mean over a larger
        # vectorized minibatch; this preserves the training objective while
        # avoiding thousands of single-sample PyTorch calls.
        loss = (
            torch.relu(-g1 + eta).mean()
            + torch.relu(-g2 + eta).mean()
            + torch.relu(-g3 + eta).mean()
            + lip_reg * network_lipschitz_tensor(net)
        )
        loss.backward()
        opt.step()

        if ep % 200 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    # Formal finite-grid check on *all* discretization representatives.
    # This is the finite premise required by the NCC theorem; random
    # sampled tests are not used to claim certification.  Keep this
    # vectorized: the g3 grid is O(|X0| |X|^2), and one-by-one PyTorch
    # calls are prohibitively slow.
    max_lprime = 0.0
    min_g1 = float("inf")
    min_g2 = float("inf")
    min_g3 = float("inf")
    chunk = 200_000

    def update_stats(g: torch.Tensor, current_min: float):
        relu_vals = torch.relu(-g + eta)
        return (
            max(max_lprime, float(relu_vals.max().item()) if relu_vals.numel() else 0.0),
            min(current_min, float(g.min().item()) if g.numel() else float("inf")),
        )

    with torch.no_grad():
        for x in trans_xs:
            fx = f(x)
            for q in q_states:
                for qp in delta(q, x):
                    g1 = net(t_input(x, q, fx, qp))[0:1]
                    rel = torch.relu(-g1 + eta)
                    max_lprime = max(max_lprime, float(rel.max().item()))
                    min_g1 = min(min_g1, float(g1.min().item()))

                    # g2
                    rows = []
                    for y in xs:
                        for j in q_states:
                            rows.append((x, float(q), y, float(j), fx, float(qp)))
                    for start_idx in range(0, len(rows), chunk):
                        part = rows[start_idx:start_idx + chunk]
                        inp1 = torch.tensor([[r[0], r[1], r[2], r[3]] for r in part], dtype=torch.float32, device=DEVICE)
                        inp2 = torch.tensor([[r[4], r[5], r[2], r[3]] for r in part], dtype=torch.float32, device=DEVICE)
                        g2 = net(inp1) - net(inp2)
                        rel = torch.relu(-g2 + eta)
                        max_lprime = max(max_lprime, float(rel.max().item()))
                        min_g2 = min(min_g2, float(g2.min().item()))

        # g3
        for x0 in x0s:
            rows = []
            for y in xs:
                for yp in xs:
                    # q_acc is [0] for this benchmark, but keep the loops
                    # explicit to mirror the theorem quantification.
                    for j in q_acc:
                        for jp in q_acc:
                            rows.append((x0, y, float(j), yp, float(jp)))
            for start_idx in range(0, len(rows), chunk):
                part = rows[start_idx:start_idx + chunk]
                inp_x0_y = torch.tensor([[r[0], 1.0, r[1], r[2]] for r in part], dtype=torch.float32, device=DEVICE)
                inp_x0_yp = torch.tensor([[r[0], 1.0, r[3], r[4]] for r in part], dtype=torch.float32, device=DEVICE)
                inp_y_yp = torch.tensor([[r[1], r[2], r[3], r[4]] for r in part], dtype=torch.float32, device=DEVICE)
                g3 = (1 - lam1) * net(inp_x0_y) - net(inp_x0_yp) - lam2 * net(inp_y_yp)
                rel = torch.relu(-g3 + eta)
                max_lprime = max(max_lprime, float(rel.max().item()))
                min_g3 = min(min_g3, float(g3.min().item()))

    lt = network_lipschitz(net)
    lf = estimate_lf_ex1()
    l1 = lt * (1.0 + lf)
    l2 = 2.0 * lt * (1.0 + lf)
    l3 = (2.0 + lam1 + lam2) * lt
    l_all = max(l1, l2, l3)

    theorem_margin = l_all * xi / 2.0 - eta
    certified = (max_lprime <= tol) and (theorem_margin <= 0.0)

    payload = {
        'arch': '4-80-1',
        'xi': xi,
        'transition_guard': 'domain_restricted: x in X and f(x) in X',
        'num_grid_points': len(xs),
        'num_transition_grid_points': len(trans_xs),
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
        'min_g1_grid': min_g1,
        'min_g2_grid': min_g2,
        'min_g3_grid': min_g3,
        'certification_sample_check': 'exhaustive_grid',
        'theorem_margin': theorem_margin,
        'certified': certified,
        'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}
    }
    try:
        Path('saved_models').mkdir(exist_ok=True)
        Path('saved_models/model.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        Path('res.txt').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'warning: skipped auxiliary saved_models write: {exc}')

    print('saved to saved_models/model.json')
    print(f'certified={certified}, theorem_margin={theorem_margin:.6f}, max_lprime={max_lprime:.6f}')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='res_ncc_ex1.json')
    p.add_argument('--max-iter', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grid-step', '--xi', dest='xi', type=float, default=0.02)
    p.add_argument('--dreal-precision', type=float, default=0.0, help='unused (kept for CLI consistency)')
    p.add_argument('--z3-timeout-ms', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--seed', type=int, default=0, help='random seed for NCC training')
    p.add_argument('--qi', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--qj', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--eta', type=float, default=0.01)
    p.add_argument('--lambda1', type=float, default=0.1)
    p.add_argument('--lambda2', type=float, default=0.1)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--lip-reg', type=float, default=1e-2, help='Lipschitz regularization weight for NCC training')
    args = p.parse_args()
    set_seed(args.seed)
    print_header("ex1", "NCC", "neural_closure_certificate", {"epochs": args.epochs, "lr": args.lr, "xi": args.xi, "seed": args.seed, "lip_reg": args.lip_reg})
    t0 = time.time()
    payload = train(epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol, seed=args.seed, lip_reg=args.lip_reg)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex1"
    payload["method"] = "NCC"
    payload["certificate_type"] = "neural_closure_certificate"
    payload["backend"] = {"train": "pytorch", "certify": "lipschitz_bound"}
    payload["automaton"] = {"states": [0, 1], "initial_state": 1, "accepting_states": [0]}
    payload["seed"] = args.seed
    payload["elapsed_sec"] = elapsed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
