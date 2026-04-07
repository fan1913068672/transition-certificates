import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time
import json
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

"""
 Kuramoto
LTL : G ¬Xu ()
NBA : F Xu ()
- q = 0NBA
- NBA : q = 1 ()

:  (Transition Persistence Certificate)
:  + CEGIS
-  B(x, q)
-
-  dReal SMT
-
"""
PI = 3.1415926
class BarrierNetwork(nn.Module):
    """ B(x, q) """
    def __init__(self, input_dim=2, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, q):
        """
        :
            x:  (tensor)
            q:  (tensor)
        :
            B(x, q):
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)

        if x.dim() == 0:
            x = x.unsqueeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)

        inp = torch.cat([x.unsqueeze(-1), q.unsqueeze(-1)], dim=-1)
        h = self.fc1(inp)
        h = self.relu(h)
        out = self.fc2(h)
        return out.squeeze(-1)

    def save_model(self, filepath):
        """"""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"✓ : {filepath}")

    def save_parameters_json(self, filepath):
        """JSON"""
        params = {
            'W1': self.fc1.weight.detach().numpy().tolist(),
            'b1': self.fc1.bias.detach().numpy().tolist(),
            'W2': self.fc2.weight.detach().numpy().tolist(),
            'b2': self.fc2.bias.detach().numpy().tolist(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'architecture': f"{self.input_dim} -> {self.hidden_dim} -> 1",
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ JSON: {filepath}")

    def save_parameters_txt(self, filepath):
        """"""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("\n")
            f.write(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f": {self.input_dim} -> {self.hidden_dim} -> 1\n")
            f.write(f": ReLU\n\n")

            f.write(" W1 (10x2):\n")
            W1 = self.fc1.weight.detach().numpy()
            for i in range(W1.shape[0]):
                f.write(f"  W1[{i}, :] = [{W1[i, 0]:.6f}, {W1[i, 1]:.6f}]\n")

            f.write("\n b1 (10x1):\n")
            b1 = self.fc1.bias.detach().numpy()
            for i in range(b1.shape[0]):
                f.write(f"  b1[{i}] = {b1[i]:.6f}\n")

            f.write("\n W2 (1x10):\n")
            W2 = self.fc2.weight.detach().numpy()
            f.write("  W2[0, :] = [")
            for i in range(W2.shape[1]):
                f.write(f"{W2[0, i]:.6f}")
                if i < W2.shape[1] - 1:
                    f.write(", ")
            f.write("]\n")

            f.write(f"\n b2 (1x1):\n")
            f.write(f"  b2 = {self.fc2.bias.detach().numpy()[0]:.6f}\n\n")

            f.write("Analytical expression:\n")
            f.write("  B(x, q) = b2 + sum_i W2[0,i] * ReLU(W1[i,0]*x + W1[i,1]*q + b1[i])\n\n")

            f.write("Python:\n")
            f.write("  def B(x, q):\n")
            f.write("      W1 = np.array(...)\n")
            f.write("      b1 = np.array(...)\n")
            f.write("      W2 = np.array(...)\n")
            f.write("      b2 = ...\n")
            f.write("      h_linear = W1[:,0]*x + W1[:,1]*q + b1\n")
            f.write("      h_relu = np.maximum(0, h_linear)\n")
            f.write("      return b2 + np.sum(W2[0,:] * h_relu)\n")

        print(f"✓ : {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """"""
        checkpoint = torch.load(filepath)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✓  {filepath} ")
        return model

def in_state_space(x_ce):
    """: x ∈ [0, 2π] (dreal)"""
    return dreal.And(x_ce >= 0, x_ce <= PI * 2)

def in_initial_set(x):
    """: x ∈ [0, π/9] ()"""
    return 4*PI/9 <= x <= PI*5 / 9

def in_initial_set_dreal(x_ce):
    """ (dreal)"""
    return dreal.And(x_ce >= 4*PI/9, x_ce <= PI*5 / 9)

def system_dynamics(x_ce):
    """ (dreal)"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x_ce + Ts * Omega + Ts * K * dreal.sin(-x_ce) - 0.532 * x_ce ** 2 + 1.69

def system_dynamics_numeric(x):
    """ ()"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x + Ts * Omega + Ts * K * math.sin(-x) - 0.532 * x ** 2 + 1.69

def in_unsafe_set(x_ce):
    """: x ∈ [7π/9, 8π/9] (dreal)"""
    return dreal.And(x_ce >= PI / 9 * 7, x_ce <= PI / 9 * 8)

def in_unsafe_set_numeric(x):
    """ ()"""
    return 7 * PI / 9 <= x <= 8 * PI / 9

def get_next_modes(q):
    """"""
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise ValueError(f": {q}")

def compute_mode_transition(x, q):
    """"""
    if q == 1:
        if in_unsafe_set_numeric(x):
            return [0]
        else:
            return [1]
    else:
        return [0]

def generate_samples(start, end, step):
    """"""
    res = []
    for i in range(int(start * int(1/step)), int(end * int(1/step)) + 1):
        res.append(i * step)
    return res

def cartesian_product(set1, set2):
    """"""
    def connect(x1, x2):
        if not isinstance(x1, list) and not isinstance(x2, list):
            return [x1, x2]
        elif isinstance(x1, list) and not isinstance(x2, list):
            return x1 + [x2]
        elif not isinstance(x1, list) and isinstance(x2, list):
            return [x1] + x2
        else:
            return x1 + x2

    if not set1 or not set2:
        return []

    res = []
    for x1 in set1:
        for x2 in set2:
            res.append(connect(x1, x2))
    return res

def multi_cartesian_product(set1, *args):
    """"""
    res = cartesian_product(set1, args[0])
    for sp in args[1:]:
        res = cartesian_product(res, sp)
    return res

def convert_network_to_dreal(B_net):
    """
    dReal
    B(x, q) = W2 @ (W1 @ [x, q] + b1) + b2
    """
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def barrier_dreal(x, q):
        """dReal ( ReLU)"""
        expr = b2[0]
        for i in range(len(b1)):

            h_linear = W1[i, 0] * x + W1[i, 1] * q + b1[i]
            cond = (h_linear > 0)
            if isinstance(cond, (bool, np.bool_)):
                # Numeric path (x, q are Python floats/ints).
                h_relu = h_linear if cond else 0.0
            else:
                # Symbolic dReal path.
                h_relu = dreal.if_then_else(cond, h_linear, 0.0)

            expr += W2[0, i] * h_relu
        return expr

    return barrier_dreal

def train_barrier_network(B_net, training_data, epochs=1000, lr=0.01):
    """

    training_data:  'init', 'unsafe', 'transition'
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        if training_data['init']:
            init_loss = 0.0
            for x0, q0 in training_data['init']:
                B_init = B_net(torch.tensor(x0, dtype=torch.float32),
                              torch.tensor(q0, dtype=torch.float32))
                init_loss += torch.relu(-B_init + 0.01)
            loss_total += init_loss / len(training_data['init'])

        if training_data['unsafe']:
            unsafe_loss = 0.0
            for xu, qu in training_data['unsafe']:
                B_unsafe = B_net(torch.tensor(xu, dtype=torch.float32),
                                torch.tensor(qu, dtype=torch.float32))
                unsafe_loss += torch.relu(B_unsafe + 0.01)
            loss_total += unsafe_loss / len(training_data['unsafe'])

        if training_data['transition']:
            trans_loss = 0.0
            for x, q, xp, qp in training_data['transition']:
                B_curr = B_net(torch.tensor(x, dtype=torch.float32),
                              torch.tensor(q, dtype=torch.float32))
                B_next = B_net(torch.tensor(xp, dtype=torch.float32),
                              torch.tensor(qp, dtype=torch.float32))
                trans_loss += torch.relu(-(B_next - B_curr))
            loss_total += trans_loss / len(training_data['transition'])

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")

def verify_barrier_with_dreal(B_net):
    """
    dReal
    : (, )
    """

    solver = dreal.Context()
    solver.config.precision = 0.0001
    solver.SetLogic(dreal.Logic.QF_NRA)
    x_var = dreal.Variable('x')
    solver.DeclareVariable(x_var, 0, 2 * PI)

    barrier_dreal = convert_network_to_dreal(B_net)
    counterexamples = {'init': [], 'unsafe': [], 'transition': []}
    has_counterexample = False

    solver.Push(2)
    solver.Assert(in_initial_set_dreal(x_var))
    solver.Assert(barrier_dreal(x_var, 1) < 0)
    model = solver.CheckSat()
    if model is not None:
        x_val = model[x_var].mid()
        print(f"  : x={x_val:.6f}, B(x,1)={barrier_dreal(x_val, 1)}")
        counterexamples['init'].append((float(x_val), 1))
        has_counterexample = True
    else:
        print("  ✓ ")
    solver.Pop(2)

    solver.Push(2)
    solver.Assert(in_state_space(x_var))
    solver.Assert(barrier_dreal(x_var, 0) >= 0)
    model = solver.CheckSat()
    if model is not None:
        x_val = model[x_var].mid()
        print(f"  : x={x_val:.6f}, B(x,0)={barrier_dreal(x_val, 0)}")
        counterexamples['unsafe'].append((float(x_val), 0))
        has_counterexample = True
    else:
        print("  ✓ ")
    solver.Pop(2)

    has_transition_counterexample = False
    solver.Push(3)
    solver.Assert(in_state_space(x_var))
    solver.Assert(in_state_space(system_dynamics(x_var)))
    solver.Assert(barrier_dreal(x_var, 1) >= 0)

    for next_q in get_next_modes(1):
        solver.Push(2)
        if next_q == 0:
            solver.Assert(in_unsafe_set(x_var))
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)
        else:  # next_q == 1
            solver.Assert(dreal.Not(in_unsafe_set(x_var)))
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)

        model = solver.CheckSat()
        if model is not None:
            x_val = model[x_var].mid()
            xp_val = system_dynamics_numeric(x_val)
            print(f"   (q=1->q={next_q}): x={x_val:.6f}, x'={xp_val:.6f}")
            counterexamples['transition'].append((float(x_val), 1, float(xp_val), next_q))
            has_counterexample = True
            has_transition_counterexample = True
        solver.Pop(2)
    solver.Pop(3)

    solver.Push(3)
    solver.Assert(in_state_space(x_var))
    solver.Assert(in_state_space(system_dynamics(x_var)))
    solver.Assert(barrier_dreal(x_var, 0) >= 0)

    for next_q in get_next_modes(0):
        solver.Push(1)
        if next_q == 0:
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)
        else:
            raise ValueError(f": 0->{next_q}")

        model = solver.CheckSat()
        if model is not None:
            x_val = model[x_var].mid()
            xp_val = system_dynamics_numeric(x_val)
            print(f"   (q=0->q={next_q}): x={x_val:.6f}, x'={xp_val:.6f}")
            counterexamples['transition'].append((float(x_val), 0, float(xp_val), next_q))
            has_counterexample = True
            has_transition_counterexample = True
        solver.Pop(1)
    solver.Pop(3)

    if not has_transition_counterexample:
        print("  ✓ ")

    return (not has_counterexample, counterexamples)

def synthesize_barrier_certificate(save_dir="saved_models", max_iterations=20, train_epochs=1000, train_lr=0.01):
    """CEGIS"""
    print("")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"barrier_net_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    B_net = BarrierNetwork(input_dim=2, hidden_dim=10)

    x_samples = generate_samples(0, 2 * PI, 1)
    q_samples = [0, 1]

    print(f"\n:")
    print(f"  x_samples:  = {len(x_samples)}")
    print(f"  q_samples:  = {len(q_samples)}")

    all_states = multi_cartesian_product(x_samples, q_samples)
    print(f"  all_states:  = {len(all_states)}")

    x0_samples = generate_samples(PI*4 / 9, PI*5 / 9, 0.1)
    print(f"  x0_samples:  = {len(x0_samples)}")

    initial_states = multi_cartesian_product(x0_samples, [1])
    print(f"  initial_states:  = {len(initial_states)}")

    unsafe_states = multi_cartesian_product(x_samples, [0])
    print(f"  unsafe_states:  = {len(unsafe_states)}")

    transition_count = 0
    for x, q in all_states:
        qp_list = compute_mode_transition(x, q)
        transition_count += len(qp_list)
    print(f"  transition_samples:  = {transition_count}")
    print(f"  : {model_dir}")
    print("-" * 40)

    training_data = {
        'init': initial_states.copy(),
        'unsafe': unsafe_states.copy(),
        'transition': []
    }

    for x, q in all_states:
        xp = system_dynamics_numeric(x)
        qp_list = compute_mode_transition(x, q)
        for qp in qp_list:
            training_data['transition'].append((x, q, xp, qp))

    iteration = 0
    success = False

    while iteration < max_iterations:
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(":")
        print(f"  init:  = {len(training_data['init'])}")
        print(f"  unsafe:  = {len(training_data['unsafe'])}")
        print(f"  transition:  = {len(training_data['transition'])}")

        print("\n1: ...")
        train_barrier_network(B_net, training_data, epochs=train_epochs, lr=train_lr)

        print("\n2: dReal...")
        verified, counterexamples = verify_barrier_with_dreal(B_net)

        if verified:
            print(f"\n{'='*60}")
            print("✓ ! ")
            print("=" * 60)
            success = True
            break
        else:

            print("\n3: ...")
            added_count = 0
            for ce in counterexamples['init']:
                training_data['init'].append(ce)
                added_count += 1
            for ce in counterexamples['unsafe']:
                training_data['unsafe'].append(ce)
                added_count += 1
            for ce in counterexamples['transition']:
                training_data['transition'].append(ce)
                added_count += 1
            print(f"   {added_count} ")
            print(f"  :")
            print(f"    : {len(counterexamples['init'])} ")
            print(f"    : {len(counterexamples['unsafe'])} ")
            print(f"    : {len(counterexamples['transition'])} ")

        iteration += 1

    if iteration >= max_iterations:
        print(f"\n{'='*60}")
        print("✗ ")
        print("=" * 60)
    elif not success:
        print(f"\n{'='*60}")
        print("✗ ")
        print("=" * 60)

    return B_net, success, model_dir, iteration, max_iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ex1 NNT synthesis")
    parser.add_argument("--out", type=str, default="res_nnt_ex1.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qi", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qj", type=int, default=0, help="unused (kept for CLI consistency)")
    args = parser.parse_args()

    print_header(
        "ex1",
        "NNT",
        "transition_safety",
        {"solver_verify": "dreal", "hidden": 10, "max_iter": args.max_iter, "epochs": args.epochs, "lr": args.lr},
    )
    start_time = time.time()
    barrier_net, success, model_dir, iteration, max_iterations = synthesize_barrier_certificate(
        max_iterations=args.max_iter,
        train_epochs=args.epochs,
        train_lr=args.lr,
    )
    end_time = time.time()

    if success:
        print(f"\n: {end_time - start_time:.4f} ")
        print("\n:")
        print("-" * 50)
        print(": 2 -> 10 -> 1 (ReLU )")
        print("\n (W1):")
        print(barrier_net.fc1.weight.detach().numpy())
        print("\n (b1):")
        print(barrier_net.fc1.bias.detach().numpy())
        print("\n (W2):")
        print(barrier_net.fc2.weight.detach().numpy())
        print("\n (b2):")
        print(barrier_net.fc2.bias.detach().numpy())
        print("-" * 50)
        print(f"\n: B(x, q) = W2 @ ReLU(W1 @ [x, q] + b1) + b2")

        print("\n" + "=" * 60)
        print("...")
        print("=" * 60)

        model_path = os.path.join(model_dir, "barrier_net.pth")
        barrier_net.save_model(model_path)

    elapsed = end_time - start_time
    result = {
        "example": "ex1",
        "method": "NNT",
        "certificate_type": "transition_safety",
        "success": bool(success),
        "iterations": int(iteration),
        "max_iterations": int(max_iterations),
        "elapsed_sec": float(elapsed),
        "solver": {"synth": "adam", "verify": "dreal"},
        "hidden_dim": int(barrier_net.fc1.out_features) if barrier_net is not None else None,
        "model_dir": model_dir,
        "model_state_path": os.path.join(model_dir, "barrier_net.pth") if success else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(success), int(iteration), elapsed, str(out_path))

