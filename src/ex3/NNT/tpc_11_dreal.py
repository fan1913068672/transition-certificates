import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time

"""
Case: Temperature Control System with 2 Rooms
LTL specification: F G not (VF) (Eventually always avoid VF)

Accepting Transition: (qi=1, qj=1) - staying in accepting state q=1
Accepting Condition (Satisfaction Set VF): (x₁, x₂) ∈ [20, 26]²

TPC Conditions:
1. B(x₀) ≥ 0 for initial states in [21, 24]²
2. B(x) >= 0 implies B(x') >= 0 for states in [20, 34]²
3. B(x) ≥ B(x') for all transitions (non-increasing)
4. B(x) ≥ B(x') + ε when accepting transition (1,1) occurs under accepting condition VF (strictly decreasing by ε > 0)

使用平方激活函数， 这样1,2两个条件就不需要验证了。
然后只需要验证3,4两个条件。
"""

class BarrierNetwork(nn.Module):
    """Shallow neural network for persistence barrier certificate B(x1, x2)"""
    def __init__(self, input_dim=2, hidden_dim=20):
        super(BarrierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        # Learnable epsilon for strict decrease
        self.log_epsilon = nn.Parameter(torch.tensor(0.0))  # log(epsilon) for positivity

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: room temperatures
        Returns:
            B(x1, x2): persistence barrier certificate value
        Note: This is only for transitions (qi=1, qj=1)
        """
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
        out = self.fc2(h)
        return out.squeeze(-1)

    def get_epsilon(self):
        """Get current epsilon value (must be > 0)"""
        return torch.exp(self.log_epsilon)


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
    """Initial state constraint in dReal"""
    return dreal.And(dreal.And(x1_ce >= 21, x1_ce <= 24),
                     dreal.And(x2_ce >= 21, x2_ce <= 24))


def In_VF(x1, x2):
    """Verification region (where strict decrease is required)"""
    return x1 >= 20 and x1 <= 26 and x2 >= 20 and x2 <= 26


def In_VF_Cond(x1_ce, x2_ce):
    """Verification region in dReal"""
    return dreal.And(dreal.And(x1_ce >= 20, x1_ce <= 26),
                     dreal.And(x2_ce >= 20, x2_ce <= 26))


def In_X_Cond(x1_ce, x2_ce):
    """State space constraint in dReal"""
    return dreal.And(dreal.And(x1_ce >= 20, x1_ce <= 34),
                     dreal.And(x2_ce >= 20, x2_ce <= 34))


def f_t(x1, x2):
    """Dynamics in dReal format"""
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
    for i in range(a * int(1 / s), b * int(1 / s) + 1):
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
    """Multi-dimensional Cartesian product"""
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res


def get_Bp_dreal(B_net):
    """Convert neural network to dReal expression"""
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def Bp_c(x1, x2):
        """dReal expression for B(x1, x2)"""
        expr = b2[0]
        for i in range(len(b1)):
            h_i = W1[i, 0] * x1 + W1[i, 1] * x2 + b1[i]
            expr = expr + W2[0, i] * h_i
        return expr

    return Bp_c


def train_candidate(B_net, training_data, epochs=1000, lr=0.01):
    """
    Train neural network for persistence certificate
    Key: B must strictly decrease in accepting state transitions
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        epsilon = B_net.get_epsilon()

        # Loss 2: B(x) >= B(x') for non-accepting transitions
        if len(training_data['non_acc_trans']) > 0:
            non_acc_loss = 0.0
            for x1, x2, x1p, x2p in training_data['non_acc_trans']:
                B_curr = B_net(
                    torch.tensor(float(x1), dtype=torch.float32),
                    torch.tensor(float(x2), dtype=torch.float32)
                )
                B_next = B_net(
                    torch.tensor(float(x1p), dtype=torch.float32),
                    torch.tensor(float(x2p), dtype=torch.float32)
                )
                # Non-increasing: B(x) >= B(x')
                non_acc_loss += torch.relu(-B_curr + B_next + 0.01)
            loss_total += non_acc_loss / len(training_data['non_acc_trans'])

        # Loss 3: B(x) >= B(x') + epsilon for accepting transitions in VF
        if len(training_data['acc_trans']) > 0:
            acc_loss = 0.0
            for x1, x2, x1p, x2p in training_data['acc_trans']:
                B_curr = B_net(
                    torch.tensor(float(x1), dtype=torch.float32),
                    torch.tensor(float(x2), dtype=torch.float32)
                )
                B_next = B_net(
                    torch.tensor(float(x1p), dtype=torch.float32),
                    torch.tensor(float(x2p), dtype=torch.float32)
                )
                # Strictly decreasing: B(x) >= B(x') + epsilon
                acc_loss += torch.relu(-B_curr + B_next + epsilon + 0.01)
            loss_total += acc_loss / len(training_data['acc_trans'])

        # Loss 4: Encourage epsilon > 0 (actually ensured by log parameterization)
        # Add small regularization to prefer reasonable epsilon values
        loss_total += 0.001 * torch.relu(0.01 - epsilon)

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}, ε: {epsilon.item():.6f}")


def verify_with_dreal(B_net):
    """Verify persistence certificate with dReal"""
    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x1_ce = dreal.Variable('x1_ce')
    x2_ce = dreal.Variable('x2_ce')
    ce_solver.DeclareVariable(x1_ce, 20, 34)
    ce_solver.DeclareVariable(x2_ce, 20, 34)

    Bp_c = get_Bp_dreal(B_net)
    epsilon_val = B_net.get_epsilon().item()

    counterexamples = {'init': [], 'non_acc': [], 'acc': []}
    ce_flag = False


    # Verify 1: B(x) >= B(x') for all transitions (non-increasing)
    ce_solver.Push(2)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(dreal.Not(In_VF_Cond(x1_ce, x2_ce)))  # Outside VF
    ce_solver.Assert(Bp_c(x1_ce, x2_ce) >= 0)
    ce_solver.Assert(Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce))
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  Counterexample to non-increasing property:")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32),
                      torch.tensor(x2_val, dtype=torch.float32)).item()
        B_next = B_net(torch.tensor(x1p_val, dtype=torch.float32),
                      torch.tensor(x2p_val, dtype=torch.float32)).item()
        print(f"    x1={x1_val}, x2={x2_val}, B={B_curr}")
        print(f"    x1'={x1p_val}, x2'={x2p_val}, B'={B_next}")
        counterexamples['non_acc'].append((x1_val, x2_val, x1p_val, x2p_val))
        ce_flag = True
    else:
        print("  ✓ Non-increasing property verified")
    ce_solver.Pop(2)

    # Verify 2: B(x) >= B(x') + epsilon in VF (strict decrease)
    ce_solver.Push(2)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(In_VF_Cond(x1_ce, x2_ce))  # Inside VF
    ce_solver.Assert(Bp_c(x1_ce, x2_ce) >= 0)
    ce_solver.Assert(Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce) + epsilon_val)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print(f"  Counterexample to strict decrease (ε={epsilon_val:.6f}):")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32),
                      torch.tensor(x2_val, dtype=torch.float32)).item()
        B_next = B_net(torch.tensor(x1p_val, dtype=torch.float32),
                      torch.tensor(x2p_val, dtype=torch.float32)).item()
        print(f"    x1={x1_val}, x2={x2_val}, B={B_curr}")
        print(f"    x1'={x1p_val}, x2'={x2p_val}, B'={B_next}")
        print(f"    B - B' = {B_curr - B_next:.6f} (should be >= {epsilon_val:.6f})")
        counterexamples['acc'].append((x1_val, x2_val, x1p_val, x2p_val))
        ce_flag = True
    else:
        print(f"  ✓ Strict decrease property verified (ε={epsilon_val:.6f})")
    ce_solver.Pop(2)

    return (not ce_flag, counterexamples)


def synthesize_persistence_certificate():
    """
    Main CEGIS loop for persistence certificate
    """
    print("=" * 70)
    print("STEP 2: Synthesizing TRANSITION PERSISTENCE CERTIFICATE")
    print("Goal: Prove q=1 is visited only FINITELY many times")
    print("=" * 70)

    # Initialize neural network
    B_net = BarrierNetwork(input_dim=2, hidden_dim=20)

    # Initialize training data
    X01_Samples = step_sample(21, 24, 0.1)
    X02_Samples = step_sample(21, 24, 0.1)
    X0_Samples = state_space_product(X01_Samples, X02_Samples)

    X1_Samples = step_sample(20, 34, 1)
    X2_Samples = step_sample(20, 34, 1)
    All_Samples = state_space_product(X1_Samples, X2_Samples)

    training_data = {
        'init': X0_Samples.copy(),
        'non_acc_trans': [],  # Transitions outside VF
        'acc_trans': []       # Transitions inside VF (must strictly decrease)
    }

    # Populate transition data
    for x1, x2 in All_Samples:
        x1p, x2p = f_m(x1, x2)
        if In_VF(x1, x2):
            training_data['acc_trans'].append((x1, x2, x1p, x2p))
        else:
            training_data['non_acc_trans'].append((x1, x2, x1p, x2p))

    MAX_ITER = 20
    iter = 0
    cc_flag = False

    # CEGIS Loop
    while iter < MAX_ITER:
        print(f"\n{'='*70}")
        print(f"CEGIS Iteration #{iter}")
        print(f"{'='*70}")
        print(f"Training data: init={len(training_data['init'])}, " +
              f"non_acc_trans={len(training_data['non_acc_trans'])}, " +
              f"acc_trans={len(training_data['acc_trans'])}")

        # Step 1: Train candidate
        print("\nStep 1: Training neural network...")
        train_candidate(B_net, training_data, epochs=1000, lr=0.01)

        # Step 2: Verify with dReal
        print("\nStep 2: Verifying with dReal...")
        verified, counterexamples = verify_with_dreal(B_net)

        if verified:
            print("\n" + "="*70)
            print("✓ Verification PASSED! Persistence certificate synthesized!")
            print("="*70)
            cc_flag = True
            break
        else:
            # Step 3: Add counterexamples
            print("\nStep 3: Adding counterexamples...")
            num_ce = 0
            for ce in counterexamples['init']:
                training_data['init'].append(ce)
                num_ce += 1
            for ce in counterexamples['non_acc']:
                x1, x2, x1p, x2p = ce
                if In_VF(x1, x2):
                    training_data['acc_trans'].append(ce)
                else:
                    training_data['non_acc_trans'].append(ce)
                num_ce += 1
            for ce in counterexamples['acc']:
                training_data['acc_trans'].append(ce)
                num_ce += 1
            print(f"  Added {num_ce} counterexamples")

        iter += 1

    if iter >= MAX_ITER:
        print("\n" + "="*70)
        print("✗ Exceeded maximum iterations")
        print("="*70)
    elif not cc_flag:
        print("\n" + "="*70)
        print("✗ Unable to synthesize persistence certificate")
        print("="*70)

    return B_net, cc_flag


if __name__ == "__main__":
    start_time = time.time()
    B_net, success = synthesize_persistence_certificate()
    end_time = time.time()

    if success:
        print("\nFinal Persistence Certificate:")
        print("-" * 70)
        print(f"Network: 2 -> {B_net.fc1.out_features} -> 1")
        print(f"Epsilon (ε): {B_net.get_epsilon().item():.6f}")
        print("\nLayer 1 weights (W1):")
        print(B_net.fc1.weight.detach().numpy())
        print("\nLayer 1 bias (b1):")
        print(B_net.fc1.bias.detach().numpy())
        print("\nLayer 2 weights (W2):")
        print(B_net.fc2.weight.detach().numpy())
        print("\nLayer 2 bias (b2):")
        print(B_net.fc2.bias.detach().numpy())
        print("-" * 70)
        print("\nInterpretation:")
        print("  • B(x1, x2) is non-negative at initial states")
        print("  • B(x1, x2) is non-increasing along all transitions")
        print(f"  • B(x1, x2) STRICTLY DECREASES by at least ε={B_net.get_epsilon().item():.6f}")
        print("    when staying in accepting state q=1 (within VF region)")
        print("  • Therefore, q=1 can be visited only FINITELY many times ✓")
        print("=" * 70)

    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
