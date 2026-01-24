import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time

"""
Case: One-dimensional Kuramoto Oscillator
LTL specification: G ¬Xu (Always avoid unsafe region)
NBA negation: F Xu (Eventually reach unsafe region)
- NBA accepting state: q = 0 (represents entering unsafe region)
- NBA initial state: q = 1 (safe state)

Verification Goal: TRANSITION PERSISTENCE CERTIFICATE for transition (0, 0)
===========================================================================
Accepting Transition: (qi=0, qj=0) - staying in accepting state
Accepting Condition (Satisfaction Set): All states X ∈ [0, 2π]

Persistence Property to Prove:
- The accepting transition (0, 0) occurs only FINITELY many times under accepting condition
- In this case: 0 times, because q=0 is never reached from initial state q=1
- This proves the NBA is not accepted (no infinite run visiting q=0 infinitely often)

Barrier Certificate Conditions:
1. B(x₀, q₀=1) ≥ 0 for initial states
2. B(x, q=0) < 0 for all x (accepting states must be negative)
3. B(x, q) ≥ 0 ⟹ B(x', q') ≥ 0 (invariant preservation)

Since B(x, 0) < 0 always and B starts ≥ 0, the system never reaches q=0,
thus transition (0,0) occurs 0 times (finitely many).

Method: Neural Network Template with CEGIS
- Shallow network B(x, q) as barrier certificate
- Trained via gradient descent with constraint-based loss
- Verified with dReal SMT solver
- Counterexample-guided iterative refinement
"""

class BarrierNetwork(nn.Module):
    """Shallow neural network for barrier certificate B(x, q)"""
    def __init__(self, input_dim=2, hidden_dim=10):
        super(BarrierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x, q):
        """
        Args:
            x: state variable (tensor)
            q: automaton state (tensor)
        Returns:
            B(x, q): barrier certificate value
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
        h = self.fc1(inp)  # Linear activation for analytical expression
        out = self.fc2(h)
        return out.squeeze(-1)


def In_X_Cond(x_ce):
    return dreal.And(x_ce >= 0, x_ce <= dreal.cos(0) * 2)


def In_X0_Cond(x_ce):
    return dreal.And(x_ce >= 0, x_ce <= dreal.cos(0) / 9)


def f_t(x_ce):
    """Dynamics in dReal format"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x_ce + Ts * Omega + Ts * K * dreal.sin(-x_ce) - 0.532 * x_ce ** 2 + 1.69


def f_m(x):
    """Dynamics in math format"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    xp = x + Ts * Omega + Ts * K * math.sin(-x) - 0.532 * x ** 2 + 1.69
    return xp


def In_Unsafe_Cond(x_ce):
    return dreal.And(x_ce >= dreal.cos(0) / 9 * 7, x_ce <= dreal.cos(0) / 9 * 8)


def q_trans(q):
    """Automaton transitions"""
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise Exception


def In_Unsafe(x):
    return x >= 7 / 9 * math.pi and x <= 8 / 9 * math.pi


def delta(x, q):
    """Automaton state transitions"""
    if q == 1:
        if In_Unsafe(x):
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
    B(x, q) = W2 @ (W1 @ [x, q] + b1) + b2
    """
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def Bp_c(x, q):
        """dReal expression for B(x, q)"""
        # h = W1 @ [x, q] + b1
        # out = W2 @ h + b2
        expr = b2[0]
        for i in range(len(b1)):
            h_i = W1[i, 0] * x + W1[i, 1] * q + b1[i]
            expr = expr + W2[0, i] * h_i
        return expr

    return Bp_c


def train_candidate(B_net, training_data, epochs=1000, lr=0.01):
    """
    Train neural network on training data

    Args:
        B_net: barrier network
        training_data: dict with keys 'init', 'unsafe', 'transition'
                       each contains list of (x, q) or (x, q, xp, qp) tuples
        epochs: number of training epochs
        lr: learning rate
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # Loss 1: B(x0, q0) >= 0 for initial states
        if len(training_data['init']) > 0:
            init_loss = 0.0
            for x0, q0 in training_data['init']:
                B_init = B_net(torch.tensor(x0), torch.tensor(q0))
                init_loss += torch.relu(-B_init + 0.01)  # penalize B < 0
            loss_total += init_loss / len(training_data['init'])

        # Loss 2: B(xu, qu) < 0 for unsafe states
        if len(training_data['unsafe']) > 0:
            unsafe_loss = 0.0
            for xu, qu in training_data['unsafe']:
                B_unsafe = B_net(torch.tensor(xu), torch.tensor(qu))
                unsafe_loss += torch.relu(B_unsafe + 0.01)  # penalize B >= 0
            loss_total += unsafe_loss / len(training_data['unsafe'])

        # Loss 3: B(x, q) >= 0 => B(x', q') >= 0 for transitions
        if len(training_data['transition']) > 0:
            trans_loss = 0.0
            for x, q, xp, qp in training_data['transition']:
                B_curr = B_net(torch.tensor(x), torch.tensor(q))
                B_next = B_net(torch.tensor(xp), torch.tensor(qp))
                # If B(x, q) >= 0, then B(x', q') should >= 0
                trans_loss += torch.relu(B_curr) * torch.relu(-B_next + 0.01)
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
    x_ce = dreal.Variable('x_ce')
    ce_solver.DeclareVariable(x_ce, 0, 2 * dreal.cos(0))

    Bp_c = get_Bp_dreal(B_net)
    counterexamples = {'init': [], 'unsafe': [], 'transition': []}
    ce_flag = False

    # Verify 1: B(x0, q0) >= 0 for initial states
    ce_solver.Push(2)
    ce_solver.Assert(In_X0_Cond(x_ce))
    ce_solver.Assert(Bp_c(x_ce, 1) < 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  Counterexample to non-negativity in initial state:")
        x_val = ce_model[x_ce].mid()
        print(f"    x={x_val}, B(x, 1)={B_net(torch.tensor(x_val), torch.tensor(1.0)).item()}")
        counterexamples['init'].append((x_val, 1))
        ce_flag = True
    else:
        print("  ✓ Initial state condition verified")
    ce_solver.Pop(2)

    # Verify 2: B(xu, qu) < 0 for unsafe states
    ce_solver.Push(2)
    ce_solver.Assert(In_X_Cond(x_ce))
    ce_solver.Assert(Bp_c(x_ce, 0) >= 0)
    ce_model = ce_solver.CheckSat()
    if ce_model != None:
        print("  Counterexample to negativity in unsafe state:")
        x_val = ce_model[x_ce].mid()
        print(f"    x={x_val}, B(x, 0)={B_net(torch.tensor(x_val), torch.tensor(0.0)).item()}")
        counterexamples['unsafe'].append((x_val, 0))
        ce_flag = True
    else:
        print("  ✓ Unsafe state condition verified")
    ce_solver.Pop(2)

    # Verify 3: Transition property for q=1
    tnn_flag = False
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x_ce))
    ce_solver.Assert(In_X_Cond(f_t(x_ce)))
    ce_solver.Assert(Bp_c(x_ce, 1) >= 0)
    qp_list = q_trans(1)
    for qp in qp_list:
        ce_solver.Push(2)
        if qp == 0:
            ce_solver.Assert(In_Unsafe_Cond(x_ce))
            ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)
        elif qp == 1:
            ce_solver.Assert(dreal.Not(In_Unsafe_Cond(x_ce)))
            ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)
        else:
            raise Exception
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print(f"  Counterexample to transition property (q=1 -> q'={qp}):")
            x_val = ce_model[x_ce].mid()
            xp_val = f_m(x_val)
            B_curr = B_net(torch.tensor(x_val), torch.tensor(1.0)).item()
            B_next = B_net(torch.tensor(xp_val), torch.tensor(float(qp))).item()
            print(f"    x={x_val}, x'={xp_val}, B(x,1)={B_curr}, B(x',{qp})={B_next}")
            counterexamples['transition'].append((x_val, 1, xp_val, qp))
            ce_flag = True
            tnn_flag = True
        ce_solver.Pop(2)
    ce_solver.Pop(3)

    # Verify 4: Transition property for q=0
    ce_solver.Push(3)
    ce_solver.Assert(In_X_Cond(x_ce))
    ce_solver.Assert(In_X_Cond(f_t(x_ce)))
    ce_solver.Assert(Bp_c(x_ce, 0) >= 0)
    qp_list = q_trans(0)
    for qp in qp_list:
        ce_solver.Push(1)
        if qp == 0:
            ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)
        else:
            raise Exception
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print(f"  Counterexample to transition property (q=0 -> q'={qp}):")
            x_val = ce_model[x_ce].mid()
            xp_val = f_m(x_val)
            B_curr = B_net(torch.tensor(x_val), torch.tensor(0.0)).item()
            B_next = B_net(torch.tensor(xp_val), torch.tensor(float(qp))).item()
            print(f"    x={x_val}, x'={xp_val}, B(x,0)={B_curr}, B(x',{qp})={B_next}")
            counterexamples['transition'].append((x_val, 0, xp_val, qp))
            ce_flag = True
            tnn_flag = True
        ce_solver.Pop(1)
    ce_solver.Pop(3)

    if not tnn_flag:
        print("  ✓ Transition property verified")

    return (not ce_flag, counterexamples)


def synthesize_barrier_certificate():
    """
    Main CEGIS loop for synthesizing barrier certificate
    """
    print("Synthesizing a state safety certificate using Neural Network Template")
    print("=" * 70)

    # Initialize neural network
    B_net = BarrierNetwork(input_dim=2, hidden_dim=10)

    # Initialize training data with samples
    X_Samples = step_sample(0, 2 * math.pi, 0.01)
    Q_Samples = [0, 1]
    Y_Samples = state_space_product(X_Samples, Q_Samples)
    X0_Samples = step_sample(0, math.pi / 9, 0.01)
    Q0_Samples = [1]
    Y0_Samples = state_space_product(X0_Samples, Q0_Samples)
    Qacc_Samples = [0]
    Yu_Samples = state_space_product(X_Samples, Qacc_Samples)

    training_data = {
        'init': Y0_Samples.copy(),
        'unsafe': Yu_Samples.copy(),
        'transition': []
    }

    # Add transition samples
    for x, q in Y_Samples:
        xp = f_m(x)
        qp_list = delta(x, q)
        for qp in qp_list:
            training_data['transition'].append((x, q, xp, qp))

    MAX_ITER = 20
    iter = 0
    cc_flag = False

    # CEGIS Loop
    while iter < MAX_ITER:
        print(f"\n{'='*70}")
        print(f"CEGIS Iteration #{iter}")
        print(f"{'='*70}")
        print(f"Training data size: init={len(training_data['init'])}, " +
              f"unsafe={len(training_data['unsafe'])}, " +
              f"transition={len(training_data['transition'])}")

        # Step 1: Train candidate barrier certificate
        print("\nStep 1: Training neural network...")
        train_candidate(B_net, training_data, epochs=1000, lr=0.01)

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

    if iter >= MAX_ITER:
        print("\n" + "="*70)
        print("✗ Exceeded maximum number of iterations")
        print("="*70)
    elif not cc_flag:
        print("\n" + "="*70)
        print("✗ Unable to synthesize barrier certificate")
        print("="*70)

    return B_net, cc_flag


if __name__ == "__main__":
    start_time = time.time()
    B_net, success = synthesize_barrier_certificate()
    end_time = time.time()

    if success:
        print("\nFinal Barrier Certificate:")
        print("-" * 70)
        print("Network architecture: 2 -> 10 -> 1 (shallow linear network)")
        print("\nLayer 1 weights (W1):")
        print(B_net.fc1.weight.detach().numpy())
        print("\nLayer 1 bias (b1):")
        print(B_net.fc1.bias.detach().numpy())
        print("\nLayer 2 weights (W2):")
        print(B_net.fc2.weight.detach().numpy())
        print("\nLayer 2 bias (b2):")
        print(B_net.fc2.bias.detach().numpy())
        print("-" * 70)
        print(f"\nAnalytical expression: B(x, q) = W2 @ (W1 @ [x, q] + b1) + b2")

    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
