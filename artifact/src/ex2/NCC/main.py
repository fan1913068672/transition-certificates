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


def in_x(x1, x2):
    return 0.0 <= x1 <= 8 * PI / 9 and 0.0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0.0 <= x1 <= PI / 9 and 0.0 <= x2 <= PI / 9


def in_xu(x1, x2):
    return (5 * PI / 6 <= x1 <= 8 * PI / 9) or (5 * PI / 6 <= x2 <= 8 * PI / 9)


def f(x1, x2):
    ts, omega, k = 0.1, 0.01, 0.0006
    return (
        x1 + ts * omega + 1.69 + ts * k * math.sin(x2 - x1) - 0.532 * ts * x1 * x1,
        x2 + ts * omega + 1.69 + ts * k * math.sin(x1 - x2) - 0.532 * ts * x2 * x2,
    )


def transition_enabled(x1, x2):
    # Domain-restricted transition semantics: only transitions with source
    # and successor in X are part of the verification problem.
    fx1, fx2 = f(x1, x2)
    return in_x(x1, x2) and in_x(fx1, fx2)


def delta(q, x1, x2):
    lab = label_of(x1, x2)
    if q == 1:
        if lab == "a":
            return [0]
        return [1]
    if q == 0:
        return [0]
    raise ValueError(f"invalid automaton state: {q}")


def label_of(x1, x2):
    if in_xu(x1, x2):
        return "a"
    return "n"


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
    return torch.tensor([[x1, x2, float(q), y1, y2, float(j)]], dtype=torch.float32, device=DEVICE)


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


CONTINUOUS_COLS = [0, 1, 3, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def network_lipschitz_tensor(net: Net):
    # q and j are discrete automaton states and are exhaustively enumerated.
    # Only continuous x/y coordinates contribute to the finite-cell radius.
    return torch.linalg.svdvals(net.fc1.weight[:, CONTINUOUS_COLS]).max() * torch.linalg.svdvals(net.fc2.weight).max()


def network_lipschitz(net: Net) -> float:
    with torch.no_grad():
        return spectral_norm(net.fc1.weight.detach()[:, CONTINUOUS_COLS]) * spectral_norm(net.fc2.weight.detach())


def estimate_lf_ex2() -> float:
    # Conservative infinity-norm Jacobian row-sum bound for the corrected
    # 2D Kuramoto dynamics
    #   f_i = x_i + ts*omega + 1.69 + ts*k*sin(x_j-x_i)
    #         - 0.532*ts*x_i^2.
    # On X=[0,8*pi/9]^2 we have
    # |df_i/dx_i| + |df_i/dx_j|
    # = |1 - ts*k*cos(.) - 1.064*ts*x_i| + |ts*k*cos(.)|
    # <= 1 + 2*ts*k, since the first absolute-value argument remains positive.
    return 1.0 + 2.0 * 0.1 * 0.0006


def train(epochs=2000, lr=1e-3, xi=0.01, eta=0.01, cert_eta=None, lam1=0.1, lam2=0.1, tol=1e-4, seed=0, lip_reg=1e-2, g2_weight=1.0):
    xi_eff = xi
    xs = []
    # AP boundary for the unsafe label a.
    for a, b in zip([0.0, 5 * PI / 6], [5 * PI / 6, 8 * PI / 9]):
        xs.extend(segment_centers(a, b, xi_eff))
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    trans_pts = [(x1, x2) for x1, x2 in pts if transition_enabled(x1, x2)]
    x0_axis = segment_centers(0.0, PI / 9, xi_eff)
    x0 = [(x1, x2) for x1 in x0_axis for x2 in x0_axis if in_x0(x1, x2)]
    q_states = [0, 1]
    q_acc = [0]

    net = Net(hidden=80).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)

    batch_size = 2048
    for ep in range(epochs):
        opt.zero_grad()
        batch = []
        for _ in range(batch_size):
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
            fx1, fx2 = f(x1, x2)
            batch.append((x1, x2, float(q), float(qp), fx1, fx2, y1, y2, float(j2), yp1, yp2, x01, x02, float(j3), float(jp)))

        inp_x_fx = torch.tensor([[r[0], r[1], r[2], r[4], r[5], r[3]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x_y = torch.tensor([[r[0], r[1], r[2], r[6], r[7], r[8]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_fx_y = torch.tensor([[r[4], r[5], r[3], r[6], r[7], r[8]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x0_y = torch.tensor([[r[11], r[12], 1.0, r[6], r[7], r[13]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_x0_yp = torch.tensor([[r[11], r[12], 1.0, r[9], r[10], r[14]] for r in batch], dtype=torch.float32, device=DEVICE)
        inp_y_yp = torch.tensor([[r[6], r[7], r[13], r[9], r[10], r[14]] for r in batch], dtype=torch.float32, device=DEVICE)

        g1 = net(inp_x_fx)
        g2 = net(inp_x_y) - net(inp_fx_y)
        g3 = (1 - lam1) * net(inp_x0_y) - net(inp_x0_yp) - lam2 * net(inp_y_yp)
        loss = (
            torch.relu(-g1 + eta).mean()
            + g2_weight * torch.relu(-g2 + eta).mean()
            + torch.relu(-g3 + eta).mean()
            + lip_reg * network_lipschitz_tensor(net)
        )
        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    max_lprime = 0.0
    min_g1 = float("inf")
    min_g2 = float("inf")
    min_g3 = float("inf")
    chunk = 200_000
    with torch.no_grad():
        for x1, x2 in trans_pts:
            for q in q_states:
                for qp in delta(q, x1, x2):
                    fx1, fx2 = f(x1, x2)
                    g1 = net(t_input(x1, x2, q, fx1, fx2, qp))[0:1]
                    rel = torch.relu(-g1 + eta)
                    min_g1 = min(min_g1, float(g1.min().item()))
                    max_lprime = max(max_lprime, float(rel.max().item()))

                    rows = []
                    for y1, y2 in pts:
                        for j in q_states:
                            rows.append((y1, y2, float(j)))
                    for start_idx in range(0, len(rows), chunk):
                        part = rows[start_idx:start_idx + chunk]
                        inp1 = torch.tensor([[x1, x2, float(q), r[0], r[1], r[2]] for r in part], dtype=torch.float32, device=DEVICE)
                        inp2 = torch.tensor([[fx1, fx2, float(qp), r[0], r[1], r[2]] for r in part], dtype=torch.float32, device=DEVICE)
                        g2 = net(inp1) - net(inp2)
                        rel = torch.relu(-g2 + eta)
                        min_g2 = min(min_g2, float(g2.min().item()))
                        max_lprime = max(max_lprime, float(rel.max().item()))

        for x01, x02 in x0:
            rows = []
            for y1, y2 in pts:
                for yp1, yp2 in pts:
                    for j in q_acc:
                        for jp in q_acc:
                            rows.append((y1, y2, float(j), yp1, yp2, float(jp)))
            for start_idx in range(0, len(rows), chunk):
                part = rows[start_idx:start_idx + chunk]
                inp_x0_y = torch.tensor([[x01, x02, 1.0, r[0], r[1], r[2]] for r in part], dtype=torch.float32, device=DEVICE)
                inp_x0_yp = torch.tensor([[x01, x02, 1.0, r[3], r[4], r[5]] for r in part], dtype=torch.float32, device=DEVICE)
                inp_y_yp = torch.tensor([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in part], dtype=torch.float32, device=DEVICE)
                g3 = (1 - lam1) * net(inp_x0_y) - net(inp_x0_yp) - lam2 * net(inp_y_yp)
                rel = torch.relu(-g3 + eta)
                min_g3 = min(min_g3, float(g3.min().item()))
                max_lprime = max(max_lprime, float(rel.max().item()))

    lt = network_lipschitz(net)
    lf = estimate_lf_ex2()
    l1 = lt * (1.0 + lf)
    l2 = 2.0 * lt * (1.0 + lf)
    l3 = (2.0 + lam1 + lam2) * lt
    l_all = max(l1, l2, l3)
    proof_eta = eta if cert_eta is None else cert_eta
    max_lprime_proof = max(0.0, proof_eta - min_g1, proof_eta - min_g2, proof_eta - min_g3)
    theorem_margin = l_all * xi_eff / 2.0 - proof_eta
    certified = (max_lprime_proof <= tol) and (theorem_margin <= 0.0)

    payload = {
        'arch': '6-80-1',
        'xi': xi,
        'xi_effective': xi_eff,
        'transition_guard': 'domain_restricted: x in X and f(x) in X',
        'num_grid_points': len(pts),
        'num_transition_grid_points': len(trans_pts),
        'eta': eta,
        'eta_train': eta,
        'eta_cert': proof_eta,
        'lambda1': lam1,
        'lambda2': lam2,
        'lip_reg': lip_reg,
        'g2_weight': g2_weight,
        'L_T': lt,
        'L_f': lf,
        'L1': l1,
        'L2': l2,
        'L3': l3,
        'L': l_all,
        'max_lprime_sampled': max_lprime_proof,
        'max_lprime_grid': max_lprime_proof,
        'max_lprime_grid_train_eta': max_lprime,
        'min_g1_grid': min_g1,
        'min_g2_grid': min_g2,
        'min_g3_grid': min_g3,
        'certification_sample_check': 'exhaustive_grid',
        'theorem_margin': theorem_margin,
        'certified': certified,
        'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()},
    }

    try:
        Path('saved_models').mkdir(exist_ok=True)
        Path('saved_models/model.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        Path('res.txt').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'warning: skipped auxiliary saved_models write: {exc}')

    print('saved to saved_models/model.json')
    print(f'certified={certified}, theorem_margin={theorem_margin:.6f}, max_lprime={max_lprime_proof:.6f}')
    return payload

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='res_ncc_ex2.json')
    p.add_argument('--max-iter', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grid-step', '--xi', dest='xi', type=float, default=0.1)
    p.add_argument('--dreal-precision', type=float, default=0.0, help='unused (kept for CLI consistency)')
    p.add_argument('--z3-timeout-ms', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--seed', type=int, default=0, help='random seed for NCC training')
    p.add_argument('--qi', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--qj', type=int, default=0, help='unused (kept for CLI consistency)')
    p.add_argument('--eta', type=float, default=0.05)
    p.add_argument('--cert-eta', type=float, default=None, help='proof margin; defaults to training eta')
    p.add_argument('--lambda1', type=float, default=0.1)
    p.add_argument('--lambda2', type=float, default=0.1)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--lip-reg', type=float, default=1e-2, help='Lipschitz regularization weight for NCC training')
    p.add_argument('--g2-weight', type=float, default=1.0, help='training weight for the NCC monotonicity/g2 loss term')
    args = p.parse_args()
    set_seed(args.seed)
    print_header("ex2", "NCC", "neural_closure_certificate", {"epochs": args.epochs, "lr": args.lr, "xi": args.xi, "seed": args.seed, "lip_reg": args.lip_reg, "g2_weight": args.g2_weight})
    t0 = time.time()
    payload = train(epochs=args.epochs, lr=args.lr, xi=args.xi, eta=args.eta, cert_eta=args.cert_eta, lam1=args.lambda1, lam2=args.lambda2, tol=args.tol, seed=args.seed, lip_reg=args.lip_reg, g2_weight=args.g2_weight)
    elapsed = time.time() - t0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    payload["example"] = "ex2"
    payload["method"] = "NCC"
    payload["certificate_type"] = "neural_closure_certificate"
    payload["backend"] = {"train": "pytorch", "certify": "lipschitz_bound"}
    payload["automaton"] = {"states": [0, 1], "initial_state": 1, "accepting_states": [0]}
    payload["seed"] = args.seed
    payload["elapsed_sec"] = elapsed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_result(bool(payload.get("certified")), 1, elapsed, str(out_path))
