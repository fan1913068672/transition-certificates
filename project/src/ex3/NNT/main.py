import math
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dreal import *
import random
import time
import numpy as np
import sympy as sp
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result

"""
Case: Temperature Control System with 2 Rooms
LTL specification: G F ¬(q=1) (Eventually always avoid accepting state)
NBA: q=1 is the accepting state of the Büchi automaton

STEP 2: Synthesize TRANSITION PERSISTENCE CERTIFICATE for transition (1, 1)
=============================================================================
Accepting Transition: (qi=1, qj=1) - staying in accepting state q=1
Accepting Condition (Satisfaction Set VF): (x₁, x₂) ∈ [20, 26]²

Persistence Property to Prove:
- The start state qi=1 of accepting transition (1, 1) is visited only FINITELY many times
  when the system satisfies the accepting condition (in VF region)
- Key mechanism: B(x) strictly DECREASES by ε > 0 when system is in VF and stays in q=1
- This proves: Transition (1,1) under accepting condition VF occurs only finitely many times
- Therefore: The NBA is not accepted (no infinite run staying in q=1 within VF infinitely)

Complementarity with Step 1:
- Step 1 (Safety): FAILS - accepting state q=1 IS reachable from initial states
- Step 2 (Persistence): SUCCEEDS - start state q=1 of accepting transition (1,1)
  is visited only finitely many times under accepting condition VF
- Conclusion: Although q=1 can be reached, the system cannot remain in q=1∩VF infinitely

Barrier Certificate Conditions:
1. B(x₀) ≥ 0 for initial states in [21, 24]²
2. B(x) ≥ B(x') for all transitions (non-increasing)
3. B(x) ≥ B(x') + ε when accepting transition (1,1) occurs under accepting condition VF
   (strictly decreasing by ε > 0)

Condition 3 ensures that the accepting transition (1,1) under accepting condition VF
can only occur finitely many times, as B is bounded and strictly decreases by ε each time.

Method: Neural Network Template with CEGIS
- B(x₁, x₂) with learnable ε parameter for strict decrease
- Note: B does not depend on q, only on continuous state (x₁, x₂)
- Trained for transition (1,1) with accepting condition VF
"""

class BarrierNetwork(nn.Module):
    """（Sum of Squares） - """
    def __init__(self, input_dim=2, hidden_dim=8):
        super(BarrierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        self.epsilon = 0.01

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)

        if x1.dim() == 0:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 0:
            x2 = x2.unsqueeze(0)

        inp = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1)

        h = self.fc1(inp)
        h = h ** 2
        out = self.fc2(h)
        return out.squeeze(-1)

    def get_epsilon(self):
        return self.epsilon

def get_Bp_dreal(B_net):
    """
    dReal

    ：
    fc1: Linear(in_features=2, out_features=hidden_dim) ->
    fc2: Linear(in_features=hidden_dim, out_features=1, bias=False)

    ：B(x) = Σ_i [W3_i * (W1_i1*x1 + W1_i2*x2 + b1_i)²]
    """

    device = next(B_net.parameters()).device

    W1 = B_net.fc1.weight.detach().cpu().numpy()  # (hidden_dim, 2)
    b1 = B_net.fc1.bias.detach().cpu().numpy()    # (hidden_dim,)
    W3 = B_net.fc2.weight.detach().cpu().numpy()  # (1, hidden_dim)

    hidden_dim = W1.shape[0]

    def Bp_c(x1, x2):
        """dReal，"""
        expr = 0.0
        for i in range(hidden_dim):

            h_i = float(W1[i, 0]) * x1 + float(W1[i, 1]) * x2 + float(b1[i])

            squared_term = h_i * h_i

            expr += float(W3[0, i]) * squared_term

        return expr

    return Bp_c

# System parameters
alpha = 0.004
theta = 0.01
Te = 0
Th = 40
mu = 0.15

def u(x_curr):
    """Control function"""
    return 0.59 - 0.011 * x_curr

def In_X0(x1, x2):
    """Initial state constraint"""
    return x1 >= 21 and x1 <= 24 and x2 >= 21 and x2 <= 24

def In_X0_Cond(x1_ce, x2_ce):
    """Initial state constraint in dreal"""
    return logical_and(logical_and(x1_ce >= 21, x1_ce <= 24),
                       logical_and(x2_ce >= 21, x2_ce <= 24))

def In_VF(x1, x2):
    """Verification region (where strict decrease is required)"""
    return x1 >= 20 and x1 <= 26 and x2 >= 20 and x2 <= 26

def In_VF_Cond(x1_ce, x2_ce):
    """Verification region in dreal"""
    return logical_and(logical_and(x1_ce >= 20, x1_ce <= 26),
                       logical_and(x2_ce >= 20, x2_ce <= 26))

def In_X_Cond(x1_ce, x2_ce):
    """State space constraint in dreal"""
    return logical_and(logical_and(x1_ce >= 20, x1_ce <= 34),
                       logical_and(x2_ce >= 20, x2_ce <= 34))

def f_cond(x1, x2):
    """Dynamics in dreal format"""
    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te
    return x1_next, x2_next

def f_m(x1, x2):
    """Dynamics in math format"""
    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te
    return x1_next, x2_next

def step_sample(a, b, s):
    """Generate sample points"""
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res

def space_product(s1, s2):
    """Cartesian product"""
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
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res

def clip_weights_nonnegative(B_net, min_val=1e-8):
    with torch.no_grad():
        B_net.fc2.weight.data = torch.clamp(B_net.fc2.weight.data, min=min_val)

def train_candidate(B_net, training_data, epochs=100, lr=0.001):
    """
    Train neural network for persistence certificate
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # Get current epsilon value
        epsilon = B_net.get_epsilon()

        # Loss 1: Non-increasing for non-accepting transitions (outside VF)
        if len(training_data['non_inc']) > 0:
            non_increasing_loss = 0.0
            for x1, x2, x1p, x2p in training_data['non_inc']:
                B_curr = B_net(
                    torch.tensor(float(x1), dtype=torch.float32),
                    torch.tensor(float(x2), dtype=torch.float32)
                )
                B_next = B_net(
                    torch.tensor(float(x1p), dtype=torch.float32),
                    torch.tensor(float(x2p), dtype=torch.float32)
                )

                violation = torch.relu(B_next - B_curr - 1e-6)
                non_increasing_loss += violation

            loss_total += 100 * non_increasing_loss / len(training_data['non_inc'])

        # Loss 2: Strict decrease for accepting transitions (inside VF)
        if len(training_data['strict_dec']) > 0:
            strict_decrease_loss = 0.0
            for x1, x2, x1p, x2p in training_data['strict_dec']:
                B_curr = B_net(
                    torch.tensor(float(x1), dtype=torch.float32),
                    torch.tensor(float(x2), dtype=torch.float32)
                )
                B_next = B_net(
                    torch.tensor(float(x1p), dtype=torch.float32),
                    torch.tensor(float(x2p), dtype=torch.float32)
                )

                violation = torch.relu(B_next - B_curr + epsilon - 1e-6)
                strict_decrease_loss += violation

            loss_total += 100 * strict_decrease_loss / len(training_data['strict_dec'])

        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(B_net.parameters(), max_norm=1.0)
        optimizer.step()

        clip_weights_nonnegative(B_net, min_val=1e-8)
        if (epoch + 1) % 10 == 0:
            # Calculate average losses for logging
            avg_init = 0.0
            avg_non_inc = 0.0
            avg_strict_dec = 0.0

            with torch.no_grad():
                current_epsilon = B_net.get_epsilon()

                if len(training_data['non_inc']) > 0:
                    for x1, x2, x1p, x2p in training_data['non_inc']:
                        B_curr = B_net(torch.tensor(float(x1), dtype=torch.float32),
                                      torch.tensor(float(x2), dtype=torch.float32))
                        B_next = B_net(torch.tensor(float(x1p), dtype=torch.float32),
                                      torch.tensor(float(x2p), dtype=torch.float32))
                        avg_non_inc += torch.relu(B_next - B_curr).item()
                    avg_non_inc /= len(training_data['non_inc'])

                if len(training_data['strict_dec']) > 0:
                    for x1, x2, x1p, x2p in training_data['strict_dec']:
                        B_curr = B_net(torch.tensor(float(x1), dtype=torch.float32),
                                      torch.tensor(float(x2), dtype=torch.float32))
                        B_next = B_net(torch.tensor(float(x1p), dtype=torch.float32),
                                      torch.tensor(float(x2p), dtype=torch.float32))
                        avg_strict_dec += torch.relu(B_next - B_curr + current_epsilon).item()
                    avg_strict_dec /= len(training_data['strict_dec'])

            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}, " +
                  f"NonInc: {avg_non_inc:.4f}, StrictDec: {avg_strict_dec:.4f}")

def verify_with_dreal(B_net):
    """
    Verify persistence certificate with dreal
    """
    # Create dreal config
    config = Config()
    config.precision = 1e-6
    config.use_polytope = True
    x1_ce = Variable('x1_ce')
    x2_ce = Variable('x2_ce')
    x1p_ce, x2p_ce = f_cond(x1_ce, x2_ce)
    Bp_c = get_Bp_dreal(B_net)
    epsilon_val = B_net.get_epsilon()
    counterexamples = {
        'non_inc': [],
        'strict_dec': []
    }
    ce_flag = False

    # Verify 3: B(x) >= B(x') for non-accepting transitions (non-increasing)
    print(f"\n  Verifying Condition 3: B(x) >= B(x') for non-accepting transitions...")

    formula = logical_and(
        logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
        logical_and(
            logical_or(
                    Bp_c(x1_ce, x2_ce) - Bp_c(x1p_ce, x2p_ce) > config.precision,
                    Bp_c(x1_ce, x2_ce) - Bp_c(x1p_ce, x2p_ce) < -config.precision),
            Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce)
        )
    )

    result = CheckSatisfiability(formula, config)
    if result:

        box = result
        x1_val = box[x1_ce].mid()
        x2_val = box[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32),
                      torch.tensor(x2_val, dtype=torch.float32)).item()
        B_next = B_net(torch.tensor(x1p_val, dtype=torch.float32),
                      torch.tensor(x2p_val, dtype=torch.float32)).item()
        print("  ✗ Counterexample to non-increasing property:")
        print(f"    x1={x1_val:.6f}, x2={x2_val:.6f}, B={B_curr:.12e}")
        print(f"    x1'={x1p_val:.6f}, x2'={x2p_val:.6f}, B'={B_next:.12e}")
        print(f"    Violation: B - B' = {B_curr - B_next:.12e} < 0")
        counterexamples['non_inc'].append((x1_val, x2_val, x1p_val, x2p_val))
        ce_flag = True
    else:
        print("  ✓ Non-increasing property verified")

    # Verify 4: B(x) >= B(x') + ε for accepting transitions (strict decrease)
    print(f"\n  Verifying Condition 4: B(x) >= B(x') + ε for accepting transitions...")

    formula = logical_and(
        logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
        logical_and(In_VF_Cond(x1_ce, x2_ce),
                   logical_and(
                   logical_or(
                        Bp_c(x1_ce, x2_ce) - Bp_c(x1p_ce, x2p_ce) - epsilon_val > config.precision,
                        Bp_c(x1_ce, x2_ce) - Bp_c(x1p_ce, x2p_ce) - epsilon_val < -config.precision
                   ),
                   Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce) + epsilon_val))
    )

    result = CheckSatisfiability(formula, config)
    if result:

        box = result
        x1_val = box[x1_ce].mid()
        x2_val = box[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32),
                      torch.tensor(x2_val, dtype=torch.float32)).item()
        B_next = B_net(torch.tensor(x1p_val, dtype=torch.float32),
                      torch.tensor(x2p_val, dtype=torch.float32)).item()
        decrease = B_curr - B_next

        print(f"  ✗ Counterexample to strict decrease (ε={epsilon_val:.6f}):")
        print(f"    x1={x1_val:.6f}, x2={x2_val:.6f}, B={B_curr:.12e}")
        print(f"    x1'={x1p_val:.6f}, x2'={x2p_val:.6f}, B'={B_next:.12e}")
        print(f"    B - B' = {decrease:.12e} < ε = {epsilon_val:.12e}")
        counterexamples['strict_dec'].append((x1_val, x2_val, x1p_val, x2p_val))
        ce_flag = True
    else:
        print(f"  ✓ Strict decrease property verified (ε={epsilon_val:.6f})")

    return (not ce_flag, counterexamples)

def synthesize_persistence_certificate(max_iterations=1000, train_epochs=50, train_lr=1e-4):
    """
    Main CEGIS loop for persistence certificate
    """
    print("=" * 70)
    print("STEP 2: Synthesizing TRANSITION PERSISTENCE CERTIFICATE ()")
    print(": (2) -> (3) -> (1)")
    print(":  (x²)")
    print(": 4")
    print(": dReal (δ-)")
    print("=" * 70)

    B_net = BarrierNetwork(input_dim=2, hidden_dim=3)

    X1_Samples = step_sample(20, 26, 1)
    X2_Samples = step_sample(20, 26, 1)
    VF_Samples = state_space_product(X1_Samples, X2_Samples)

    X1_Samples = step_sample(26.1, 34, 1)
    X2_Samples = step_sample(26.1, 34, 1)
    NVF_Samples = state_space_product(X1_Samples, X2_Samples)

    training_data = {
        'non_inc': [],
        'strict_dec': []
    }

    for x1, x2 in VF_Samples:
        x1p, x2p = f_m(x1, x2)
        training_data['strict_dec'].append((x1, x2, x1p, x2p))

    for x1, x2 in NVF_Samples:
        x1p, x2p = f_m(x1, x2)
        training_data['non_inc'].append((x1, x2, x1p, x2p))

    strict_dec_size = len(training_data['strict_dec'])
    training_data['non_inc'] = random.sample(
        training_data['non_inc'],
        min(strict_dec_size, len(training_data['non_inc']))
    )

    iter = 0
    cc_flag = False

    while iter < max_iterations:
        print(f"\n{'='*70}")
        print(f"Iteration {iter + 1}")
        print(f"{'='*70}")
        print(f"non_inc={len(training_data['non_inc'])}, " +
              f"strict_dec={len(training_data['strict_dec'])}")

        print("\n1: ...")
        train_candidate(B_net, training_data, epochs=train_epochs, lr=train_lr)

        print("\n2: dreal...")
        verified, counterexamples = verify_with_dreal(B_net)

        if verified:
            print("\n" + "="*70)
            print("✓ ! !")
            print("="*70)
            cc_flag = True
            break
        else:

            print("\n3: ...")
            num_ce = 0

            for ce in counterexamples['non_inc']:
                training_data['non_inc'].append(ce)
                num_ce += 1

            for ce in counterexamples['strict_dec']:
                training_data['strict_dec'].append(ce)
                num_ce += 1

            print(f"   {num_ce} ")

        iter += 1

        model_path = "persistence_model_2layer.pth"
        torch.save(B_net.state_dict(), model_path)

    if iter >= max_iterations:
        print("\n" + "="*70)
        print("✗ ")
        print("="*70)
    elif not cc_flag:
        print("\n" + "="*70)
        print("✗ ")
        print("="*70)

    return B_net, cc_flag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ex3 NNT synthesis")
    parser.add_argument("--out", type=str, default="res_nnt_ex3.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grid-step", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--dreal-precision", type=float, default=0.0, help="unused (kept for CLI consistency)")
    parser.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--seed", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--qi", type=int, default=1, help="unused (kept for CLI consistency)")
    parser.add_argument("--qj", type=int, default=1, help="unused (kept for CLI consistency)")
    args = parser.parse_args()

    print_header(
        "ex3",
        "NNT",
        "transition_persistence",
        {"solver_verify": "dreal", "hidden": 4, "max_iter": args.max_iter, "epochs": args.epochs, "lr": args.lr},
    )
    start_time = time.time()
    B_net, success = synthesize_persistence_certificate(
        max_iterations=args.max_iter,
        train_epochs=args.epochs,
        train_lr=args.lr,
    )
    end_time = time.time()

    if success:
        print("\n:")
        print("-" * 70)
        print(f": 2 -> {B_net.fc1.out_features} -> 1")
        print(f":  (x²)")
        print(f": 4")
        print(f"Epsilon (ε): {B_net.get_epsilon():.6f} ()")
        print("\n W1:")
        print(B_net.fc1.weight.detach().numpy())
        print("\n b1:")
        print(B_net.fc1.bias.detach().numpy())
        print("-" * 70)
        print("-" * 70)
        print("\n:")
        print("  • B(x1, x2) > 1000  ()")
        print("  • B(x1, x2) ")
        print(f"  • B(x1, x2) VF ε={B_net.get_epsilon():.6f}")
        print("  • ，q=1  ✓")

        print("\n B  ():")
        sample_inits = [(21, 21), (21, 24), (24, 21), (24, 24), (22.5, 22.5)]
        for x1, x2 in sample_inits:
            B_val = B_net(torch.tensor(x1, dtype=torch.float32),
                         torch.tensor(x2, dtype=torch.float32)).item()
            print(f"  B({x1:.1f}, {x2:.1f}) = {B_val:.2f}")

        print("=" * 70)

        model_path = "persistence_barrier_model_2layer.pth"
        torch.save(B_net.state_dict(), model_path)
        print(f"\n✓  {model_path}")

        checkpoint_path = "persistence_barrier_checkpoint_2layer.pth"
        checkpoint = {
            'state_dict': B_net.state_dict(),
            'hidden_dim': B_net.fc1.out_features,
            'epsilon': B_net.get_epsilon(),
            'W1': B_net.fc1.weight.detach().numpy(),
            'b1': B_net.fc1.bias.detach().numpy(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓  {checkpoint_path}")

    elapsed = end_time - start_time
    print(f"\nElapsed time: {elapsed:.4f} s")

    result = {
        "example": "ex3",
        "method": "NNT",
        "certificate_type": "transition_persistence",
        "success": bool(success),
        "elapsed_sec": float(elapsed),
        "epsilon": float(B_net.get_epsilon()) if B_net is not None else None,
        "hidden_dim": int(B_net.fc1.out_features) if B_net is not None else None,
        "solver": {"synth": "adam", "verify": "dreal"},
        "model_state_path": "persistence_barrier_model_2layer.pth" if success else None,
        "checkpoint_path": "persistence_barrier_checkpoint_2layer.pth" if success else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(success), None, elapsed, str(out_path))
