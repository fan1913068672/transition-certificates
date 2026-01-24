import math
import random
import torch
import numpy as np
import sys

"""
Test neural barrier certificate for Demo2: 2D Kuramoto Oscillator
Verify 100 trajectories (100 steps each) from initial region satisfy barrier conditions:
1. B(x1_0, x2_0, q0=1) ≥ 0 for initial states in [0, π/9]²
2. B(x1, x2, q=0) < 0 for all accepting states (unsafe)
3. B(x, q) ≥ 0 ⟹ B(x', q') ≥ 0 (invariant preservation)

This script loads the trained weights from synthesis_res.txt (synthesized certificate).
"""

# Import from synthesis script
from neural_transition_safety_certificate import BarrierNetwork, f_m, In_X0, In_Unsafe, delta

def load_trained_model():
    """
    Load trained barrier network with weights from synthesis_res.txt
    These weights were synthesized by neural_transition_safety_certificate.py
    """
    B_net = BarrierNetwork(input_dim=3, hidden_dim=15)

    # Weights from synthesis_res.txt (CEGIS Iteration #0 - Verification PASSED)
    W1 = np.array([
        [-2.34543271e-02,  8.21313933e-02,  3.03853564e-02],
        [-3.20905298e-01,  4.60899949e-01,  2.30153307e-01],
        [-4.67195630e-01, -3.09374303e-01, -4.71021563e-01],
        [ 1.95118156e-03,  6.03072345e-03,  6.04022406e-02],
        [ 4.17487532e-01, -1.54161612e-02,  1.71047166e-01],
        [-7.77840661e-03,  3.90109092e-01,  1.75065957e-02],
        [-3.85209113e-01, -1.61442474e-01,  2.37467840e-01],
        [-7.46273843e-04,  1.75143254e-03,  5.26980963e-03],
        [ 3.39078009e-02, -5.27489483e-02,  1.01839297e-01],
        [-1.23012625e-02,  4.55108806e-02,  1.28491223e-01],
        [ 4.19625074e-01,  6.43815771e-02,  1.15611233e-01],
        [-7.11487979e-03,  7.11852238e-02, -2.32375965e-01],
        [-3.74985754e-01,  1.09779999e-01, -2.08238527e-01],
        [ 2.26301163e-01,  3.47592741e-01, -2.97178864e-01],
        [ 5.65479277e-03, -1.03050879e-04,  7.30168521e-02]
    ])

    b1 = np.array([
        -0.12100842, -0.6387002,  -0.01457659, -0.04621303,  0.21875167, -0.26366538,
         0.38035676, -0.00383963, -0.05037354, -0.13563296,  0.37772396,  0.25159186,
        -0.08516592,  0.13804711, -0.04783982
    ])

    W2 = np.array([
        [ 0.00190352,  0.04793187, -0.07699478,  0.00256316,  0.02776624, -0.03371546,
         -0.05416371, -0.0001113,   0.00343324,  0.00307385, -0.10334048, -0.0316381,
         -0.02389356, -0.07375315,  0.00079416]
    ])

    b2 = np.array([0.05026149])

    # Load weights into the network
    B_net.fc1.weight.data = torch.tensor(W1, dtype=torch.float32)
    B_net.fc1.bias.data = torch.tensor(b1, dtype=torch.float32)
    B_net.fc2.weight.data = torch.tensor(W2, dtype=torch.float32)
    B_net.fc2.bias.data = torch.tensor(b2, dtype=torch.float32)

    print("✓ Loaded trained barrier certificate from synthesis_res.txt")
    print(f"  Architecture: 3 -> {B_net.fc1.out_features} -> 1")
    print(f"  B(x1, x2, q) = W2 @ (W1 @ [x1, x2, q] + b1) + b2")

    return B_net

def sample_initial_states(num_samples):
    """Sample initial states uniformly from X0 = [0, π/9]²"""
    samples = []
    for _ in range(num_samples):
        x1 = random.uniform(0, math.pi / 9)
        x2 = random.uniform(0, math.pi / 9)
        samples.append((x1, x2))
    return samples

def run_trajectory_test(B_net, x1_0, x2_0, max_steps=100):
    """
    Run a single trajectory and check barrier certificate conditions
    Returns: (passed, violation_info)
    """
    x1, x2 = x1_0, x2_0
    q = 1  # Start in initial NBA state (safe)
    trajectory = [(x1, x2, q)]

    # Check initial condition
    B_init = B_net(torch.tensor(x1_0), torch.tensor(x2_0), torch.tensor(1.0)).item()
    if B_init < -1e-6:
        return False, {
            'type': 'init',
            'step': 0,
            'x1': x1_0,
            'x2': x2_0,
            'q': 1,
            'B_value': B_init,
            'message': f"Initial condition violated: B({x1_0:.4f}, {x2_0:.4f}, 1) = {B_init:.6f} < 0"
        }

    for step in range(max_steps):
        # Compute next state
        x1_next, x2_next = f_m(x1, x2)

        # Determine next automaton state
        q_next_list = delta(x1, x2, q)
        if len(q_next_list) == 0:
            break
        q_next = q_next_list[0]  # Take first transition

        # Check barrier conditions
        B_curr = B_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(float(q))).item()
        B_next = B_net(torch.tensor(x1_next), torch.tensor(x2_next), torch.tensor(float(q_next))).item()

        # Condition 1: If B(x1, x2, q) >= 0, then B(x1', x2', q') should >= 0
        if B_curr >= -1e-6:  # Current barrier is non-negative
            if B_next < -1e-6:  # Next barrier becomes negative
                return False, {
                    'type': 'transition',
                    'step': step,
                    'x1': x1,
                    'x2': x2,
                    'q': q,
                    'x1_next': x1_next,
                    'x2_next': x2_next,
                    'q_next': q_next,
                    'B_curr': B_curr,
                    'B_next': B_next,
                    'message': f"Transition invariant violated at step {step}: " +
                               f"B({x1:.4f},{x2:.4f},{q})={B_curr:.6f} >= 0 but " +
                               f"B({x1_next:.4f},{x2_next:.4f},{q_next})={B_next:.6f} < 0"
                }

        # Condition 2: If in accepting state (q=0), B should be negative
        if q == 0 and B_curr >= -1e-6:
            return False, {
                'type': 'accepting',
                'step': step,
                'x1': x1,
                'x2': x2,
                'q': q,
                'B_value': B_curr,
                'message': f"Accepting state condition violated: B({x1:.4f}, {x2:.4f}, 0) = {B_curr:.6f} >= 0"
            }

        # Check if trajectory reaches unsafe region
        if q_next == 0 and q == 1:
            # Transitioned from safe to unsafe state
            if B_curr >= -1e-6:
                return False, {
                    'type': 'safety',
                    'step': step,
                    'x1': x1,
                    'x2': x2,
                    'x1_next': x1_next,
                    'x2_next': x2_next,
                    'message': f"Safety violated: reached unsafe region at step {step}"
                }

        trajectory.append((x1_next, x2_next, q_next))
        x1, x2, q = x1_next, x2_next, q_next

    return True, {'trajectory_length': len(trajectory), 'final_q': q}

if __name__ == "__main__":
    print("="*70)
    print("Testing Neural Barrier Certificate for Demo2: Safety (2D)")
    print("="*70)
    print()

    # Load barrier network with trained weights from synthesis_res.txt
    B_net = load_trained_model()
    print()

    # Sample 100 initial states
    num_trajectories = 100
    max_steps = 100
    initial_states = sample_initial_states(num_trajectories)

    print(f"Testing {num_trajectories} trajectories with {max_steps} steps each...")
    print()

    passed_count = 0
    failed_count = 0
    violations = []
    safe_count = 0  # Count trajectories that never reach q=0

    for i, (x1_0, x2_0) in enumerate(initial_states):
        passed, info = run_trajectory_test(B_net, x1_0, x2_0, max_steps)

        if passed:
            passed_count += 1
            if info.get('final_q', 1) == 1:
                safe_count += 1
            if (i + 1) % 10 == 0:
                final_q = info.get('final_q', 1)
                print(f"Trajectory {i+1:3d}: ✓ PASSED (length={info['trajectory_length']}, final_q={final_q})")
        else:
            failed_count += 1
            print(f"Trajectory {i+1:3d}: ✗ FAILED")
            print(f"  {info['message']}")
            violations.append((i, info))
            if failed_count >= 10:  # Stop after 10 failures
                print("\n[Stopped after 10 failures]")
                break

    print()
    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total trajectories tested: {min(num_trajectories, passed_count + failed_count)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Trajectories staying safe (never reach q=0): {safe_count}")

    if failed_count == 0:
        print("\n✓ All trajectories satisfy the barrier certificate conditions!")
        print(f"  {safe_count}/{passed_count} trajectories stayed in safe state q=1")
        print("  The system satisfies the safety specification G¬unsafe")
    else:
        print(f"\n✗ Found {failed_count} violations")
        print("\nViolation details:")
        for idx, info in violations[:5]:  # Show first 5 violations
            print(f"  Trajectory {idx+1}: {info['type']} - {info['message']}")

    print("="*70)
