import argparse
import json
from pathlib import Path
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

PI = 3.1415926


def in_x(x1, x2):
    return 0 <= x1 <= 8 * PI / 9 and 0 <= x2 <= 8 * PI / 9


def in_x0(x1, x2):
    return 0 <= x1 <= PI / 9 and 0 <= x2 <= PI / 9


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


def train(epochs=2000, lr=1e-3, eta=0.01):
    xs = grid(0.0, 8 * PI / 9, 0.2)
    pts = [(x1, x2) for x1 in xs for x2 in xs if in_x(x1, x2)]
    x0 = [(x1, x2) for x1, x2 in pts if in_x0(x1, x2)]
    xu = [(x1, x2) for x1, x2 in pts if in_xu(x1, x2)]

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        for x1, x2 in random.sample(pts, min(150, len(pts))):
            y1, y2 = f(x1, x2)
            if not in_x(y1, y2):
                continue
            for q in [0, 1]:
                for qp in delta(q, x1, x2):
                    v = net(torch.tensor([[x1, x2, float(q), y1, y2, float(qp)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(-v + eta)

                    z1, z2 = random.choice(pts)
                    j = random.choice([0, 1])
                    t_xz = net(torch.tensor([[x1, x2, float(q), z1, z2, float(j)]], dtype=torch.float32))[0]
                    t_yz = net(torch.tensor([[y1, y2, float(qp), z1, z2, float(j)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(t_yz - t_xz + eta)

        for x1, x2 in random.sample(x0, min(30, len(x0))):
            y1, y2 = random.choice(xu)
            t = net(torch.tensor([[x1, x2, 1.0, y1, y2, 0.0]], dtype=torch.float32))[0]
            loss = loss + torch.relu(t + eta)

        loss.backward()
        opt.step()

        if ep % 250 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    Path('saved_models').mkdir(exist_ok=True)
    payload = {'arch': '6-80-1', 'eta': eta, 'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}}
    Path('saved_models/model.json').write_text(json.dumps(payload), encoding='utf-8')
    Path('res.txt').write_text(json.dumps({'epochs': epochs, 'eta': eta}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved to saved_models/model.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eta', type=float, default=0.01)
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, eta=args.eta)
