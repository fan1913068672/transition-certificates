import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time
import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

"""
Case: Two-dimensional Coupled Kuramoto Oscillators
LTL specification: G ¬Xu (Always avoid unsafe region)
NBA negation: F Xu (Eventually reach unsafe region)
- NBA accepting state: q = 0 (represents entering unsafe region)
- NBA initial state: q = 1 (safe state)

Verification Goal: TRANSITION PERSISTENCE CERTIFICATE for transition (0, 0)
===========================================================================
Accepting Transition: (qi=0, qj=0) - staying in accepting state
Accepting Condition (Satisfaction Set): All states (x₁, x₂) ∈ [0, 8π/9]²

Persistence Property to Prove:
- The start state qi=0 of accepting transition (0, 0) is visited only FINITELY many times
  under the accepting condition
- In this case: 0 times, because q=0 is never reached from initial state q=1
- This proves the NBA is not accepted (no infinite run with transition (0,0) occurring infinitely)

Barrier Certificate Conditions:
1. B(x₁₀, x₂₀, q₀=1) ≥ 0 for initial states
2. B(x₁, x₂, q=0) < 0 for all (x₁, x₂) (accepting states must be negative)
3. B(x, q) ≥ 0 ⟹ B(x', q') ≥ 0 (invariant preservation)

Since B(x, 0) < 0 always and B starts ≥ 0, the system never reaches q=0,
thus the start state q=0 of transition (0,0) is visited 0 times (finitely many).

Method: Neural Network Template with CEGIS
- Shallow network B(x₁, x₂, q) as barrier certificate
- Trained via gradient descent with constraint-based loss
- Verified with dReal SMT solver
- Counterexample-guided iterative refinement
"""
PI = 3.1415926
class BarrierNetwork(nn.Module):
    """Shallow neural network for barrier certificate B(x1, x2, q)"""
    def __init__(self, input_dim=3, hidden_dim=15):
        super(BarrierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x1, x2, q):
        """
        Args:
            x1: state variable 1 (tensor)
            x2: state variable 2 (tensor)
            q: automaton state (tensor)
        Returns:
            B(x1, x2, q): barrier certificate value
        """
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)

        if x1.dim() == 0:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 0:
            x2 = x2.unsqueeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)

        inp = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1), q.unsqueeze(-1)], dim=-1)
        h = self.fc1(inp)
        h = self.relu(h)
        out = self.fc2(h)
        return out.squeeze(-1)

    def save_model(self, filepath):
        """"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim if hasattr(self, 'input_dim') else 3,
            'hidden_dim': self.hidden_dim if hasattr(self, 'hidden_dim') else 15
        }, filepath)
        print(f"✓ : {filepath}")

    def save_parameters_json(self, filepath):
        """JSON"""
        params = {
            'W1': self.fc1.weight.detach().numpy().tolist(),
            'b1': self.fc1.bias.detach().numpy().tolist(),
            'W2': self.fc2.weight.detach().numpy().tolist(),
            'b2': self.fc2.bias.detach().numpy().tolist(),
            'architecture': "3 -> 15 -> 1",
            'activation': "ReLU",
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ JSON: {filepath}")

    def save_parameters_txt(self, filepath):
        """"""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(" (Ex2)\n")
            f.write(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(": 3 -> 15 -> 1\n")
            f.write(": ReLU\n\n")

            f.write(" W1 (15x3):\n")
            W1 = self.fc1.weight.detach().numpy()
            for i in range(W1.shape[0]):
                f.write(f"  W1[{i}, :] = [{W1[i, 0]:.6f}, {W1[i, 1]:.6f}, {W1[i, 2]:.6f}]\n")

            f.write("\n b1 (15x1):\n")
            b1 = self.fc1.bias.detach().numpy()
            for i in range(b1.shape[0]):
                f.write(f"  b1[{i}] = {b1[i]:.6f}\n")

            f.write("\n W2 (1x15):\n")
            W2 = self.fc2.weight.detach().numpy()
            f.write("  W2[0, :] = [")
            for i in range(W2.shape[1]):
                f.write(f"{W2[0, i]:.6f}")
                if i < W2.shape[1] - 1:
                    f.write(", ")
            f.write("]\n")

            f.write(f"\n b2 (1x1):\n")
            f.write(f"  b2 = {self.fc2.bias.detach().numpy()[0]:.6f}\n\n")

            f.write(":\n")
            f.write("  B(x1, x2, q) = b2 + Σ_{i=1}^{15} W2[0,i] * ReLU(W1[i,0]*x1 + W1[i,1]*x2 + W1[i,2]*q + b1[i])\n")

        print(f"✓ : {filepath}")

def In_X_Cond(x1_ce, x2_ce):
    return dreal.And(dreal.And(x1_ce >= 0, x1_ce <= PI / 9 * 8),
                     dreal.And(x2_ce >= 0, x2_ce <= PI / 9 * 8))

def In_X0(x1, x2):
    """Initial state constraint (Python version)"""
    return x1 >= 0 and x1 <= PI / 9 and x2 >= 0 and x2 <= PI / 9

def In_X0_Cond(x1_ce, x2_ce):
    return dreal.And(dreal.And(x1_ce >= 0, x1_ce <= PI / 9),
                     dreal.And(x2_ce >= 0, x2_ce <= PI / 9))

def f_t(x1, x2):
    """Dynamics in dReal format"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega + 1.69 + Ts * K * dreal.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega + 1.69 + Ts * K * dreal.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p

def f_m(x1, x2):
    """Dynamics in math format"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega + 1.69 + Ts * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega + 1.69 + Ts * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p

def In_Unsafe(x1, x2):
    return (x1 >= 5 / 6 * PI and x1 <= 8 / 9 * PI) or \
           (x2 >= 5 / 6 * PI and x2 <= 8 / 9 * PI)

def In_Unsafe_Cond(x1_ce, x2_ce):
    return dreal.Or(dreal.And(x1_ce >= PI / 6 * 5, x1_ce <= PI / 9 * 8),
                    dreal.And(x2_ce >= PI / 6 * 5, x2_ce <= PI / 9 * 8))

def q_trans(q):
    """Automaton transitions"""
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise Exception

def delta(x1, x2, q):
    """Automaton state transitions"""
    if q == 1:
        if In_Unsafe(x1, x2):
            return [0]
        else:
            return [1]
    else:
        return [0]

def step_sample(a, b, s):
    """Generate sample points"""
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res

def space_product(s1, s2):
    """Cartesian product of two lists"""
    def conn(x1, x2):
        if type(x1) != list and type(x2) != list:
            return [x1, x2]
        elif type(x1) == list and type(x2) != list:
            return x1 + [x2]
        elif type(x1) != list and type(x2) == list:
            return [x1] + x2
        else:
            return x1 + x2

    if s1 == [] or s2 == []:
        return []
    res = []
    for x1 in s1:
        for x2 in s2:
            res.append(conn(x1, x2))
    return res

def state_space_product(s1, *args):
    """Multi-dimensional Cartesian product"""
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res

def get_Bp_dreal(B_net):
    """
    Convert neural network to dReal expression
    B(x1, x2, q) = W2 @ ReLU(W1 @ [x1, x2, q] + b1) + b2
    """
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def Bp_c(x1, x2, q):
        """dReal expression for B(x1, x2, q)"""
        expr = b2[0]
        for i in range(len(b1)):
            h_linear = W1[i, 0] * x1 + W1[i, 1] * x2 + W1[i, 2] * q + b1[i]
            h_relu = dreal.if_then_else(h_linear > 0, h_linear, 0)
            expr = expr + W2[0, i] * h_relu
        return expr

    return Bp_c

def train_candidate(B_net, training_data, epochs=1000, lr=0.01):
    """
    Train neural network on training data

    Args:
        B_net: barrier network
        training_data: dict with keys 'init', 'unsafe', 'transition'
                       each contains list of (x1, x2, q) or (x1, x2, q, x1p, x2p, qp) tuples
        epochs: number of training epochs
        lr: learning rate
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # Loss 1: B(x01, x02, q0) >= 0 for initial states
        if len(training_data['init']) > 0:
            init_loss = 0.0
            for x01, x02, q0 in training_data['init']:
                B_init = B_net(torch.tensor(x01, dtype=torch.float32), torch.tensor(x02, dtype=torch.float32), torch.tensor(q0, dtype=torch.float32))
                init_loss += torch.relu(-B_init + 0.01)  # penalize B < 0
            loss_total += init_loss / len(training_data['init'])

        # Loss 2: B(xu1, xu2, qu) < 0 for unsafe states
        if len(training_data['unsafe']) > 0:
            unsafe_loss = 0.0
            for xu1, xu2, qu in training_data['unsafe']:
                B_unsafe = B_net(torch.tensor(xu1, dtype=torch.float32), torch.tensor(xu2, dtype=torch.float32), torch.tensor(qu, dtype=torch.float32))
                unsafe_loss += torch.relu(B_unsafe + 0.01)  # penalize B >= 0
            loss_total += unsafe_loss / len(training_data['unsafe'])

        # Loss 3: B(x1, x2, q) >= 0 => B(x1', x2', q') >= 0 for transitions
        # Using S-procedure (ncc.pdf): ReLU(-(B(x',q') - B(x,q)))
        # This requires B(x',q') >= B(x,q), i.e., barrier is non-decreasing
        if len(training_data['transition']) > 0:
            trans_loss = 0.0
            for x1, x2, q, x1p, x2p, qp in training_data['transition']:
                B_curr = B_net(torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(q, dtype=torch.float32))
                B_next = B_net(torch.tensor(x1p, dtype=torch.float32), torch.tensor(x2p, dtype=torch.float32), torch.tensor(qp, dtype=torch.float32))
                # S-procedure: penalize B_next < B_curr
                trans_loss += torch.relu(-(B_next - B_curr))  # = torch.relu(B_curr - B_next)
            loss_total += trans_loss / len(training_data['transition'])

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")

def verify_with_dreal(B_net):
    """
    Verify the candidate barrier certificate using dReal
    Returns: (verified, counterexamples)
        verified: True if verified, False if counterexamples found
        counterexamples: dict with keys 'init', 'unsafe', 'transition'
    """
    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.0001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x1_ce = dreal.Variable('x1_ce')
    x2_ce = dreal.Variable('x2_ce')
    ce_solver.DeclareVariable(x1_ce, 0, PI / 9 * 8)
    ce_solver.DeclareVariable(x2_ce, 0, PI / 9 * 8)

    Bp_c = get_Bp_dreal(B_net)
    counterexamples = {'init': [], 'unsafe': [], 'transition': []}
    ce_flag = False

    # Verify 1: B(x01, x02, q0) >= 0 for initial states
    ce_solver.Push(2)
    ce_solver.Assert(In_X0_Cond(x1_ce, x2_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) < 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  Counterexample to non-negativity in initial state:")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        B_val = B_net(torch.tensor(x1_val), torch.tensor(x2_val), torch.tensor(1.0)).item()
        print(f"    x1={x1_val}, x2={x2_val}, B(x1,x2,1)={B_val}")
        counterexamples['init'].append((x1_val, x2_val, 1))
        ce_flag = True
    else:
        print("  ✓ Initial state condition verified")
    ce_solver.Pop(2)

    # Verify 2: B(xu1, xu2, qu) < 0 for unsafe states
    ce_solver.Push(2)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) >= 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  Counterexample to negativity in unsafe state:")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        B_val = B_net(torch.tensor(x1_val), torch.tensor(x2_val), torch.tensor(0.0)).item()
        print(f"    x1={x1_val}, x2={x2_val}, B(x1,x2,0)={B_val}")
        counterexamples['unsafe'].append((x1_val, x2_val, 0))
        ce_flag = True
    else:
        print("  ✓ Unsafe state condition verified")
    ce_solver.Pop(2)

    # Verify 3: Transition property for q=1
    tnn_flag = False
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) >= 0)
    qp_list = q_trans(1)
    for qp in qp_list:
        ce_solver.Push(2)
        if qp == 0:
            ce_solver.Assert(In_Unsafe_Cond(x1_ce, x2_ce))
            ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, qp) < 0)
        elif qp == 1:
            ce_solver.Assert(dreal.Not(In_Unsafe_Cond(x1_ce, x2_ce)))
            ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, qp) < 0)
        else:
            raise Exception
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print(f"  Counterexample to transition property (q=1 -> q'={qp}):")
            x1_val = ce_model[x1_ce].mid()
            x2_val = ce_model[x2_ce].mid()
            x1p_val, x2p_val = f_m(x1_val, x2_val)
            B_curr = B_net(torch.tensor(x1_val), torch.tensor(x2_val), torch.tensor(1.0)).item()
            B_next = B_net(torch.tensor(x1p_val), torch.tensor(x2p_val), torch.tensor(float(qp))).item()
            print(f"    x1={x1_val}, x2={x2_val}, x1'={x1p_val}, x2'={x2p_val}")
            print(f"    B(x1,x2,1)={B_curr}, B(x1',x2',{qp})={B_next}")
            counterexamples['transition'].append((x1_val, x2_val, 1, x1p_val, x2p_val, qp))
            ce_flag = True
            tnn_flag = True
        ce_solver.Pop(2)
    ce_solver.Pop(3)

    # Verify 4: Transition property for q=0
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) >= 0)
    qp_list = q_trans(0)
    for qp in qp_list:
        ce_solver.Push(1)
        if qp == 0:
            ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, qp) < 0)
        else:
            raise Exception
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print(f"  Counterexample to transition property (q=0 -> q'={qp}):")
            x1_val = ce_model[x1_ce].mid()
            x2_val = ce_model[x2_ce].mid()
            x1p_val, x2p_val = f_m(x1_val, x2_val)
            B_curr = B_net(torch.tensor(x1_val), torch.tensor(x2_val), torch.tensor(0.0)).item()
            B_next = B_net(torch.tensor(x1p_val), torch.tensor(x2p_val), torch.tensor(float(qp))).item()
            print(f"    x1={x1_val}, x2={x2_val}, x1'={x1p_val}, x2'={x2p_val}")
            print(f"    B(x1,x2,0)={B_curr}, B(x1',x2',{qp})={B_next}")
            counterexamples['transition'].append((x1_val, x2_val, 0, x1p_val, x2p_val, qp))
            ce_flag = True
            tnn_flag = True
        ce_solver.Pop(1)
    ce_solver.Pop(3)

    if not tnn_flag:
        print("  ✓ Transition property verified")

    return (not ce_flag, counterexamples)

def synthesize_barrier_certificate(save_dir="saved_models", max_iterations=2000, train_epochs=1000, train_lr=0.01):
    """
    Main CEGIS loop for synthesizing barrier certificate
    """
    print("Synthesizing a state safety certificate using Neural Network Template")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"barrier_net_ex2_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    B_net = BarrierNetwork(input_dim=3, hidden_dim=15)

    # Initialize training data with samples
    X1_Samples = step_sample(0, 8 * PI / 9, 1)
    X2_Samples = step_sample(0, 8 * PI / 9, 1)
    Q_Samples = [0, 1]
    Y_Samples = state_space_product(X1_Samples, X2_Samples, Q_Samples)
    X01_Samples = step_sample(0, PI / 9, 1)
    X02_Samples = step_sample(0, PI / 9, 1)
    Q0_Samples = [1]
    Y0_Samples = state_space_product(X01_Samples, X02_Samples, Q0_Samples)
    Qacc_Samples = [0]
    Yu_Samples = state_space_product(X1_Samples, X2_Samples, Qacc_Samples)

    training_data = {
        'init': Y0_Samples.copy(),
        'unsafe': Yu_Samples.copy(),
        'transition': []
    }

    # Add transition samples
    for x1, x2, q in Y_Samples:
        x1p, x2p = f_m(x1, x2)
        qp_list = delta(x1, x2, q)
        for qp in qp_list:
            training_data['transition'].append((x1, x2, q, x1p, x2p, qp))

    iter = 0
    cc_flag = False

    # CEGIS Loop
    while iter < max_iterations:
        print(f"\n{'='*70}")
        print(f"CEGIS Iteration #{iter}")
        print(f"{'='*70}")
        print(f"Training data size: init={len(training_data['init'])}, " +
              f"unsafe={len(training_data['unsafe'])}, " +
              f"transition={len(training_data['transition'])}")

        # Step 1: Train candidate barrier certificate
        print("\nStep 1: Training neural network...")
        train_candidate(B_net, training_data, epochs=train_epochs, lr=train_lr)

        # Step 2: Verify with dReal
        print("\nStep 2: Verifying with dReal...")
        verified, counterexamples = verify_with_dreal(B_net)

        if verified:
            print("\n" + "="*70)
            print("✓ Verification PASSED! Barrier certificate synthesized successfully!")
            print("="*70)
            cc_flag = True
            break
        else:
            # Step 3: Add counterexamples to training data
            print("\nStep 3: Adding counterexamples to training data...")
            num_ce = 0
            for ce in counterexamples['init']:
                training_data['init'].append(ce)
                num_ce += 1
            for ce in counterexamples['unsafe']:
                training_data['unsafe'].append(ce)
                num_ce += 1
            for ce in counterexamples['transition']:
                training_data['transition'].append(ce)
                num_ce += 1
            print(f"  Added {num_ce} counterexamples to training data")

        iter += 1

    if iter >= max_iterations:
        print("\n" + "="*70)
        print("✗ Exceeded maximum number of iterations")
        print("="*70)
    elif not cc_flag:
        print("\n" + "="*70)
        print("✗ Unable to synthesize barrier certificate")
        print("="*70)

    return B_net, cc_flag, model_dir, iter, max_iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ex2 NNT synthesis")
    parser.add_argument("--out", type=str, default="res_nnt_ex2.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=2000)
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
        "ex2",
        "NNT",
        "transition_safety",
        {"solver_verify": "dreal", "hidden": 15, "max_iter": args.max_iter, "epochs": args.epochs, "lr": args.lr},
    )
    start_time = time.time()
    B_net, success, model_dir, iter_count, max_iter = synthesize_barrier_certificate(
        max_iterations=args.max_iter,
        train_epochs=args.epochs,
        train_lr=args.lr,
    )
    end_time = time.time()

    if success:
        print("\nFinal Barrier Certificate:")
        print("-" * 70)
        print("Network architecture: 3 -> 15 -> 1 (ReLU activation network)")
        print("\nLayer 1 weights (W1):")
        print(B_net.fc1.weight.detach().numpy())
        print("\nLayer 1 bias (b1):")
        print(B_net.fc1.bias.detach().numpy())
        print("\nLayer 2 weights (W2):")
        print(B_net.fc2.weight.detach().numpy())
        print("\nLayer 2 bias (b2):")
        print(B_net.fc2.bias.detach().numpy())
        print("-" * 70)
        print(f"\nAnalytical expression: B(x1, x2, q) = W2 @ ReLU(W1 @ [x1, x2, q] + b1) + b2")

        print("\n" + "=" * 70)
        print("Saving model and parameters...")
        B_net.save_model(os.path.join(model_dir, "barrier_net.pth"))
        B_net.save_parameters_json(os.path.join(model_dir, "parameters.json"))
        B_net.save_parameters_txt(os.path.join(model_dir, "parameters.txt"))
        print("=" * 70)

    elapsed = end_time - start_time
    result = {
        "example": "ex2",
        "method": "NNT",
        "certificate_type": "transition_safety",
        "success": bool(success),
        "iterations": int(iter_count),
        "max_iterations": int(max_iter),
        "elapsed_sec": float(elapsed),
        "solver": {"synth": "adam", "verify": "dreal"},
        "hidden_dim": int(B_net.fc1.out_features) if B_net is not None else None,
        "model_dir": model_dir,
        "model_state_path": os.path.join(model_dir, "barrier_net.pth") if success else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(success), int(iter_count), elapsed, str(out_path))
