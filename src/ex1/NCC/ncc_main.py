import argparse
import json
from pathlib import Path
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

PI = 3.1415926


def in_x(x):
    return 0 <= x <= 2 * PI


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
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sample_grid(step=0.05):
    xs = []
    x = 0.0
    while x <= 2 * PI + 1e-9:
        xs.append(round(x, 8))
        x += step
    return xs


def train(epochs=1500, lr=1e-3, eta=0.01, save='res.txt'):
    xs = sample_grid(step=0.05)
    x0s = [x for x in xs if in_x0(x)]
    xus = [x for x in xs if in_xu(x)]

    net = Net(hidden=80)
    opt = optim.Adam(net.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        # C1: T((x,q),(f(x),q')) >= 0
        for x in random.sample(xs, min(120, len(xs))):
            y = f(x)
            if not in_x(y):
                continue
            for q in [0, 1]:
                for qp in delta(q, x):
                    v = net(torch.tensor([[x, float(q), y, float(qp)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(-v + eta)

        # C2: T(x,z) >= T(y,z) (sampled strengthening)
        for x in random.sample(xs, min(80, len(xs))):
            y = f(x)
            if not in_x(y):
                continue
            for q in [0, 1]:
                for qp in delta(q, x):
                    z = random.choice(xs)
                    j = random.choice([0, 1])
                    t_xz = net(torch.tensor([[x, float(q), z, float(j)]], dtype=torch.float32))[0]
                    t_yz = net(torch.tensor([[y, float(qp), z, float(j)]], dtype=torch.float32))[0]
                    loss = loss + torch.relu(t_yz - t_xz + eta)

        # C3: initial to unsafe separation
        for x0 in random.sample(x0s, min(30, len(x0s))):
            yu = random.choice(xus)
            t = net(torch.tensor([[x0, 1.0, yu, 0.0]], dtype=torch.float32))[0]
            loss = loss + torch.relu(t + eta)

        loss.backward()
        opt.step()

        if ep % 200 == 0:
            print(f'epoch={ep}, loss={float(loss.item()):.6f}')

    out = {'arch': '4-80-1', 'eta': eta, 'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}}
    Path('saved_models').mkdir(exist_ok=True)
    Path('saved_models/model.json').write_text(json.dumps(out), encoding='utf-8')
    Path(save).write_text(json.dumps({'epochs': epochs, 'eta': eta}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved to saved_models/model.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eta', type=float, default=0.01)
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, eta=args.eta)
