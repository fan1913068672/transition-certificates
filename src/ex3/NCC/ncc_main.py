import argparse
import json
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim


def in_x(x1, x2):
    return 20 <= x1 <= 34 and 20 <= x2 <= 34


def in_x0(x1, x2):
    return 21 <= x1 <= 24 and 21 <= x2 <= 24


def in_vf(x1, x2):
    return 20 <= x1 <= 26 and 20 <= x2 <= 26


def f(x1, x2):
    alpha, theta, mu, Th, Te = 0.004, 0.01, 0.15, 40.0, 0.0
    def u(x):
        return 0.59 - 0.011 * x
    return (
        (1 - 2 * alpha - theta - mu * u(x1)) * x1 + alpha * x2 + mu * Th * u(x1) + theta * Te,
        alpha * x1 + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te,
    )


def delta_main(q, x1, x2):
    if q == 0:
        return [1] if in_x0(x1, x2) else [2]
    if q == 1:
        return [1]
    return [2]


def delta_closure(q, x1, x2):
    a0, a1 = int(in_x0(x1, x2)), int(in_vf(x1, x2))
    if q == 0:
        if (a0, a1) == (1, 1):
            return [2]
        if (a0, a1) in [(1, 0), (0, 1)]:
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
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grid(a, b, step):
    vals = []
    x = a
    while x <= b + 1e-9:
        vals.append(round(x, 8))
        x += step
    return vals


def train(mode='main', epochs=2200, lr=1e-3, eta=0.01, delta=0.01):
    xs = grid(20.0, 34.0, 0.5)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    vf = [(x1, x2) for x1, x2 in pts if in_vf(x1, x2)]

    delta_fn = delta_main if mode == 'main' else delta_closure
    q_states = [0, 1, 2] if mode == 'main' else [0, 1, 2, 3]
    qacc = 1 if mode == 'main' else 2

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        # C1 + C2
        for x1, x2 in random.sample(pts, min(150, len(pts))):
            y1, y2 = f(x1, x2)
            if not in_x(y1, y2):
                continue
            for q in q_states:
                for qp in delta_fn(q, x1, x2):
                    v = net(torch.tensor([[x1, x2, float(q), y1, y2, float(qp)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(-v + eta)

                    z1, z2 = random.choice(pts)
                    j = random.choice(q_states)
                    t_xz = net(torch.tensor([[x1, x2, float(q), z1, z2, float(j)]], dtype=torch.float32))[0]
                    t_yz = net(torch.tensor([[y1, y2, float(qp), z1, z2, float(j)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(t_yz - t_xz + eta)

        # C3 persistence decrease on accepting state
        for x01, x02 in random.sample(x0, min(30, len(x0))):
            y1, y2 = random.choice(vf)
            yp1, yp2 = random.choice(vf)
            t_x0_y = net(torch.tensor([[x01, x02, 0.0, y1, y2, float(qacc)]], dtype=torch.float32))[0]
            t_x0_yp = net(torch.tensor([[x01, x02, 0.0, yp1, yp2, float(qacc)]], dtype=torch.float32))[0]
            t_y_yp = net(torch.tensor([[y1, y2, float(qacc), yp1, yp2, float(qacc)]], dtype=torch.float32))[0]
            loss = loss + torch.relu(-t_x0_y + eta)
            loss = loss + torch.relu(-t_y_yp + eta)
            loss = loss + torch.relu((t_x0_yp + delta) - t_x0_y + eta)

        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    Path('saved_models').mkdir(exist_ok=True)
    payload = {
        'arch': '6-80-1',
        'mode': mode,
        'eta': eta,
        'delta': delta,
        'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}
    }
    Path('saved_models/model.json').write_text(json.dumps(payload), encoding='utf-8')
    Path('res.txt').write_text(json.dumps({'mode': mode, 'epochs': epochs, 'eta': eta, 'delta': delta}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved to saved_models/model.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['main', 'closure'], default='main')
    p.add_argument('--epochs', type=int, default=2200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eta', type=float, default=0.01)
    p.add_argument('--delta', type=float, default=0.01)
    args = p.parse_args()
    train(mode=args.mode, epochs=args.epochs, lr=args.lr, eta=args.eta, delta=args.delta)
