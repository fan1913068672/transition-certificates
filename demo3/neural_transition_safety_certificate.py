import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time

"""
Case: Temperature Control System with 2 Rooms
LTL specification: G ¬(q=1) (Never reach accepting state)
NBA: q=1 is the accepting state

STEP 1: Attempt to synthesize TRANSITION SAFETY CERTIFICATE
Verification Goal:
- Prove that accepting state q=1 is UNREACHABLE from initial states
- This would mean transition (q=0 → q=1) never occurs
- If successful, the system never enters the accepting state q=1

Expected Result: FAIL after multiple CEGIS iterations
- The accepting state q=1 IS reachable from initial states (q=0, x ∈ [21,24]²)
- Even with counterexample refinement, safety constraints cannot be satisfied
- This demonstrates the limitation of safety certificates for this system
- Motivates Step 2: Use persistence certificate to prove finite visits instead

Method: Neural Network Template with CEGIS
- Use shallow neural network B(x1, x2, q) as barrier certificate
- Train to satisfy safety conditions
- Verify with dReal SMT solver
- Add counterexamples and iterate
- Show that constraints cannot be satisfied even after multiple iterations
"""

class BarrierNetwork(nn.Module):
    """Shallow neural network for barrier certificate B(x1, x2, q)"""
    def __init__(self, input_dim=3, hidden_dim=20):
        super(BarrierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x1, x2, q):
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
        out = self.fc2(h)
        return out.squeeze(-1)


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


def delta(x1, x2, q):
    """Automaton state transitions"""
    if q == 0:
        if In_X0(x1, x2):
            return [1]
        else:
            return [2]
    elif q == 1:
        return [1]
    elif q == 2:
        return [2]
    else:
        raise Exception("Invalid automaton state")


def step_sample(a, b, s):
    """Generate sample points"""
    res = []
    for i in range(a * int(1 / s), b * int(1 / s) + 1):
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
    """Convert neural network to dReal expression"""
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def Bp_c(x1, x2, q):
        """dReal expression for B(x1, x2, q)"""
        expr = b2[0]
        for i in range(len(b1)):
            h_i = W1[i, 0] * x1 + W1[i, 1] * x2 + W1[i, 2] * q + b1[i]
            expr = expr + W2[0, i] * h_i
        return expr

    return Bp_c


def train_candidate(B_net, training_data, epochs=1000, lr=0.01):
    """Train neural network on training data"""
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # Loss 1: B(x01, x02, q0) >= 0 for initial states
        if len(training_data['init']) > 0:
            init_loss = 0.0
            for x01, x02, q0 in training_data['init']:
                B_init = B_net(
                    torch.tensor(float(x01), dtype=torch.float32),
                    torch.tensor(float(x02), dtype=torch.float32),
                    torch.tensor(float(q0), dtype=torch.float32)
                )
                init_loss += torch.relu(-B_init + 0.01)
            loss_total += init_loss / len(training_data['init'])

        # Loss 2: B(xu1, xu2, qu) < 0 for unsafe states (q=1)
        if len(training_data['unsafe']) > 0:
            unsafe_loss = 0.0
            for xu1, xu2, qu in training_data['unsafe']:
                B_unsafe = B_net(
                    torch.tensor(float(xu1), dtype=torch.float32),
                    torch.tensor(float(xu2), dtype=torch.float32),
                    torch.tensor(float(qu), dtype=torch.float32)
                )
                unsafe_loss += torch.relu(B_unsafe + 0.01)
            loss_total += unsafe_loss / len(training_data['unsafe'])

        # Loss 3: B(x1, x2, q) >= 0 => B(x1', x2', q') >= 0 for transitions
        if len(training_data['transition']) > 0:
            trans_loss = 0.0
            for x1, x2, q, x1p, x2p, qp in training_data['transition']:
                B_curr = B_net(
                    torch.tensor(float(x1), dtype=torch.float32),
                    torch.tensor(float(x2), dtype=torch.float32),
                    torch.tensor(float(q), dtype=torch.float32)
                )
                B_next = B_net(
                    torch.tensor(float(x1p), dtype=torch.float32),
                    torch.tensor(float(x2p), dtype=torch.float32),
                    torch.tensor(float(qp), dtype=torch.float32)
                )
                trans_loss += torch.relu(B_curr) * torch.relu(-B_next + 0.01)
            loss_total += trans_loss / len(training_data['transition'])

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")


def verify_with_dreal(B_net):
    """Verify with dReal SMT solver"""
    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x1_ce = dreal.Variable('x1_ce')
    x2_ce = dreal.Variable('x2_ce')
    ce_solver.DeclareVariable(x1_ce, 20, 34)
    ce_solver.DeclareVariable(x2_ce, 20, 34)

    Bp_c = get_Bp_dreal(B_net)
    counterexamples = {'init': [], 'unsafe': [], 'transition': []}
    ce_flag = False

    # Verify 1: B(x01, x02, q0) >= 0 for initial states
    ce_solver.Push(2)
    ce_solver.Assert(In_X0_Cond(x1_ce, x2_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) < 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  ✗ Counterexample: B < 0 at initial state")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        B_val = B_net(torch.tensor(x1_val, dtype=torch.float32),
                     torch.tensor(x2_val, dtype=torch.float32),
                     torch.tensor(0.0, dtype=torch.float32)).item()
        print(f"    x1={x1_val:.4f}, x2={x2_val:.4f}, B(x1,x2,0)={B_val:.6f}")
        counterexamples['init'].append((x1_val, x2_val, 0))
        ce_flag = True
    else:
        print("  ✓ Initial state condition verified")
    ce_solver.Pop(2)

    # Verify 2: B(xu1, xu2, qu=1) < 0 for unsafe states
    ce_solver.Push(2)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) >= 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  ✗ Counterexample: B >= 0 at unsafe state q=1")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        B_val = B_net(torch.tensor(x1_val, dtype=torch.float32),
                     torch.tensor(x2_val, dtype=torch.float32),
                     torch.tensor(1.0, dtype=torch.float32)).item()
        print(f"    x1={x1_val:.4f}, x2={x2_val:.4f}, B(x1,x2,1)={B_val:.6f}")
        counterexamples['unsafe'].append((x1_val, x2_val, 1))
        ce_flag = True
    else:
        print("  ✓ Unsafe state condition verified")
    ce_solver.Pop(2)

    # Verify 3: Transition properties
    # For q=0 -> q=1 or q=2
    for qp in [1, 2]:
        ce_solver.Push(3)
        ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
        x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
        ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
        ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) >= 0)

        if qp == 1:
            ce_solver.Assert(In_X0_Cond(x1_ce, x2_ce))
        else:  # qp == 2
            ce_solver.Assert(dreal.Not(In_X0_Cond(x1_ce, x2_ce)))

        ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, qp) < 0)
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print(f"  ✗ Counterexample: transition (0 -> {qp}) violates invariant")
            x1_val = ce_model[x1_ce].mid()
            x2_val = ce_model[x2_ce].mid()
            x1p_val, x2p_val = f_m(x1_val, x2_val)
            B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32),
                          torch.tensor(x2_val, dtype=torch.float32),
                          torch.tensor(0.0, dtype=torch.float32)).item()
            B_next = B_net(torch.tensor(x1p_val, dtype=torch.float32),
                          torch.tensor(x2p_val, dtype=torch.float32),
                          torch.tensor(float(qp), dtype=torch.float32)).item()
            print(f"    B(x,0)={B_curr:.6f}, B(x',{qp})={B_next:.6f}")
            counterexamples['transition'].append((x1_val, x2_val, 0, x1p_val, x2p_val, qp))
            ce_flag = True
        ce_solver.Pop(3)

    # For q=1 -> q=1
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) >= 0)
    ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, 1) < 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print(f"  ✗ Counterexample: transition (1 -> 1) violates invariant")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        counterexamples['transition'].append((x1_val, x2_val, 1, x1p_val, x2p_val, 1))
        ce_flag = True
    ce_solver.Pop(3)

    # For q=2 -> q=2
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
    x1p_ce, x2p_ce = f_t(x1_ce, x2_ce)
    ce_solver.Assert(In_X_Cond(x1p_ce, x2p_ce))
    ce_solver.Assert(Bp_c(x1_ce, x2_ce, 2) >= 0)
    ce_solver.Assert(Bp_c(x1p_ce, x2p_ce, 2) < 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print(f"  ✗ Counterexample: transition (2 -> 2) violates invariant")
        x1_val = ce_model[x1_ce].mid()
        x2_val = ce_model[x2_ce].mid()
        x1p_val, x2p_val = f_m(x1_val, x2_val)
        counterexamples['transition'].append((x1_val, x2_val, 2, x1p_val, x2p_val, 2))
        ce_flag = True
    ce_solver.Pop(3)

    if not ce_flag:
        print("  ✓ All transition properties verified")

    return (not ce_flag, counterexamples)


def synthesize_safety_certificate():
    """
    Attempt to synthesize TRANSITION SAFETY CERTIFICATE with full CEGIS loop
    """
    print("=" * 70)
    print("STEP 1: Attempting to synthesize TRANSITION SAFETY CERTIFICATE")
    print("Goal: Prove accepting state q=1 is UNREACHABLE")
    print("=" * 70)

    # Initialize neural network
    B_net = BarrierNetwork(input_dim=3, hidden_dim=20)

    # Initialize training data
    X1_Samples = step_sample(20, 34, 1)
    X2_Samples = step_sample(20, 34, 1)
    Q_Samples = [0, 1, 2]
    Y_Samples = state_space_product(X1_Samples, X2_Samples, Q_Samples)

    X01_Samples = step_sample(21, 24, 0.1)
    X02_Samples = step_sample(21, 24, 0.1)
    Q0_Samples = [0]
    Y0_Samples = state_space_product(X01_Samples, X02_Samples, Q0_Samples)

    # For safety certificate: q=1 is the UNSAFE accepting state
    Qacc_Samples = [1]
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

    MAX_ITER = 15
    cc_flag = False

    # CEGIS Loop
    for iter in range(MAX_ITER):
        print(f"\n{'='*70}")
        print(f"CEGIS Iteration #{iter}")
        print(f"{'='*70}")
        print(f"Training data: init={len(training_data['init'])}, " +
              f"unsafe={len(training_data['unsafe'])}, " +
              f"transition={len(training_data['transition'])}")

        # Step 1: Train candidate
        print("\nStep 1: Training neural network...")
        train_candidate(B_net, training_data, epochs=1000, lr=0.01)

        # Step 2: Verify with dReal
        print("\nStep 2: Verifying with dReal...")
        verified, counterexamples = verify_with_dreal(B_net)

        if verified:
            print("\n" + "="*70)
            print("✓ Verification PASSED! (Unexpected)")
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
            for ce in counterexamples['unsafe']:
                training_data['unsafe'].append(ce)
                num_ce += 1
            for ce in counterexamples['transition']:
                training_data['transition'].append(ce)
                num_ce += 1
            print(f"  Added {num_ce} counterexamples to training data")

    print("\n" + "="*70)
    if cc_flag:
        print("✓ UNEXPECTED: Safety certificate synthesized!")
        print("="*70)
    else:
        print("✗ EXPECTED RESULT: Unable to synthesize safety certificate")
        print("="*70)
        print("\nExplanation:")
        print("  After {0} CEGIS iterations, safety certificate could not be synthesized.".format(MAX_ITER))
        print("  The accepting state q=1 IS REACHABLE from initial states:")
        print("    - Starting from q=0 with (x₁,x₂) ∈ [21,24]²")
        print("    - The automaton transitions to q=1 (accepting state)")
        print("    - Therefore, safety certificate CANNOT exist")
        print("\nConclusion:")
        print("  → Safety approach FAILS for this system")
        print("  → Need to use PERSISTENCE certificate instead!")
        print("  → See neural_transition_persistence_certificate.py for Step 2")
        print("="*70)

    return B_net, cc_flag


if __name__ == "__main__":
    start_time = time.time()
    B_net, success = synthesize_safety_certificate()
    end_time = time.time()

    print(f"\nTime taken: {end_time - start_time:.4f} seconds")

    print("\n" + "=" * 70)
    print("NEXT STEP:")
    print("  Run: python neural_transition_persistence_certificate.py")
    print("  This will synthesize a PERSISTENCE certificate to prove")
    print("  that accepting transition (1,1) under accepting condition VF")
    print("  occurs only FINITELY many times.")
    print("=" * 70)
