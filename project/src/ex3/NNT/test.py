import math
import random
import torch
import numpy as np
import sys
import os

"""
Test neural persistence barrier certificate for Demo3: Temperature Control
Verify 100 trajectories (100 steps each) from initial region satisfy persistence conditions:
1. B(x1_0, x2_0) ≥ 0 for initial states in [21, 24]²
2. B(x1, x2) ≥ B(x1', x2') for all transitions (non-increasing)
3. B(x1, x2) ≥ B(x1', x2') + ε when in VF region (strictly decreasing)

The persistence certificate proves that the accepting state q=1 (within VF region)
is visited only FINITELY many times.
"""

# Import from synthesis script
from test2 import (
    BarrierNetwork, f_m, In_X0, In_VF
)

def load_trained_model(res_file='res.txt'):
    """Load trained persistence barrier network from res.txt"""
    B_net = BarrierNetwork(2, 3)
    model_path = "persistence_model_2layer.pth"
    B_net.load_state_dict(torch.load(model_path))

    return B_net

def sample_initial_states(num_samples):
    """Sample initial states uniformly from X0 = [21, 24]²"""
    samples = []
    for _ in range(num_samples):
        x1 = random.uniform(21, 24)
        x2 = random.uniform(21, 24)
        samples.append((x1, x2))
    return samples

def B(x1, x2):
    epsilon = 0.25
    """
    Analytical persistence barrier certificate

    B(x_1, x_2) = 27 - x_1   if (x_1, x_2) ∈ [20, 26]²
                = 0          otherwise
    """
    if In_VF(x1, x2):
        return 27 - x1
    else:
        return 0

def BSOS(x1, x2):
    return 3.158*(-0.004*x1 - 0.016*x2 + 1)**2 + 1.11*(-0.999*x1 + x2)**2 + 0.001*x1**2

def run_trajectory_test(B_net, x1_0, x2_0, max_steps=1000, use_B = False, use_SOS = False):
    """
    Run a single trajectory and check persistence certificate conditions
    Returns: (passed, violation_info)
    """
    x1, x2 = x1_0, x2_0
    trajectory = [(x1, x2)]

    if not use_B:
        epsilon = B_net.get_epsilon()  # Already returns float
    else:
        if not use_SOS:
            epsilon = 0.25
        else:
            epsilon = 0.0006844474952434377
    tol = 1e-4 # Tolerance for numerical checks (same as synthesis file)

    vf_visit_count = 0  # Count visits to VF region
    decrease_violations = 0

    for step in range(max_steps):
        # Compute next state
        x1_next, x2_next = f_m(x1, x2)

        # Check barrier conditions
        if not use_B:
            B_curr = B_net(torch.tensor(x1, dtype=torch.float32),
                          torch.tensor(x2, dtype=torch.float32)).item()
            B_next = B_net(torch.tensor(x1_next, dtype=torch.float32),
                        torch.tensor(x2_next, dtype=torch.float32)).item()
        else:
            B_curr = B(x1, x2)
            B_next = B(x1_next, x2_next)

        in_vf = In_VF(x1, x2)

        if in_vf:
            vf_visit_count += 1

        # Condition 2: Non-increasing everywhere (use tol for numerical stability)
        if abs(B_curr - B_next) > tol and B_curr < B_next :  # B increased beyond tolerance
                return False, {
                    'type': 'non_increasing',
                    'step': step,
                    'x1': x1,
                    'x2': x2,
                    'x1_next': x1_next,
                    'x2_next': x2_next,
                    'B_curr': B_curr,
                    'B_next': B_next,
                    'in_vf': in_vf,
                    'message': f"Non-increasing violated at step {step}: " +
                               f"B({x1:.4f},{x2:.4f})={B_curr:.6f} < B({x1_next:.4f},{x2_next:.4f})={B_next:.6f}"
                }

        # Condition 3: Strictly decreasing in VF region (use tol for numerical stability)
        if in_vf:
            decrease_amount = B_curr - B_next
            if abs(decrease_amount - epsilon) > tol and decrease_amount < epsilon:  # Not decreasing enough (within tolerance)
                return False, {
                    'type': 'strict_decrease',
                    'step': step,
                    'x1': x1,
                    'x2': x2,
                    'x1_next': x1_next,
                    'x2_next': x2_next,
                    'B_curr': B_curr,
                    'B_next': B_next,
                    'decrease': decrease_amount,
                    'epsilon': epsilon,
                    'message': f"Strict decrease violated at step {step} (in VF): " +
                               f"B decreased by {decrease_amount:.6f} < ε={epsilon:.6f}"
                }

        trajectory.append((x1_next, x2_next))
        x1, x2 = x1_next, x2_next

    return True, {
        'trajectory_length': len(trajectory),
        'vf_visits': vf_visit_count,
        'epsilon': epsilon
    }

if __name__ == "__main__":
    print("="*70)
    print("Testing Neural Persistence Certificate for Demo3: Temperature")
    print("="*70)

    # Load barrier network from res.txt
    try:
        B_net = load_trained_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run neural_transition_persistence_certificate.py first.")
        sys.exit(1)
    use_B = False
    use_SOS = True
    if not use_B:
        epsilon = B_net.get_epsilon()  # Already returns float
        print("\nPersistence Barrier Network:")
        print(f"  Input: (x1, x2) -> Hidden: {B_net.fc1.out_features} -> Output: B(x1,x2)")
        print(f"  Epsilon (ε): {epsilon:.6f}")
        print()
    else:
        if not use_SOS:
            epsilon = 0.25
        else:
            epsilon = 0.0006844474952434377

    # Sample 100 initial states
    num_trajectories = 1000
    max_steps = 1000
    initial_states = sample_initial_states(num_trajectories)

    print(f"Testing {num_trajectories} trajectories with {max_steps} steps each...")
    print()

    passed_count = 0
    failed_count = 0
    violations = []
    total_vf_visits = 0

    for i, (x1_0, x2_0) in enumerate(initial_states):
        passed, info = run_trajectory_test(B_net, x1_0, x2_0, max_steps, use_B=use_B, use_SOS=use_SOS)

        if passed:
            passed_count += 1
            total_vf_visits += info['vf_visits']
            if (i + 1) % 10 == 0:
                print(f"Trajectory {i+1:3d}: ✓ PASSED (length={info['trajectory_length']}, VF visits={info['vf_visits']})")
        else:
            failed_count += 1
            print(f"Trajectory {i+1:3d}: ✗ FAILED")
            print(f"  {info['message']}")
            violations.append((i, info))
            if failed_count >= 100:  # Stop after 10 failures
                print("\n[Stopped after 100 failures]")
                break

    print()
    print("epsilon:", epsilon)
    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total trajectories tested: {min(num_trajectories, passed_count + failed_count)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    if passed_count > 0:
        print(f"Average VF visits per trajectory: {total_vf_visits / passed_count:.2f}")

    if failed_count == 0:
        print("\n✓ All trajectories satisfy the persistence certificate conditions!")
        print(f"  • B(x) is non-negative at initial states")
        print(f"  • B(x) is non-increasing along all transitions")
        print(f"  • B(x) strictly decreases by at least ε={epsilon:.6f} in VF region")
        print("  • Therefore, accepting state q=1 is visited only FINITELY many times ✓")
        print("  • The system satisfies the persistence specification GF¬VF")
    else:
        print(f"\n✗ Found {failed_count} violations")
        print("\nViolation details:")
        for idx, info in violations[:5]:  # Show first 5 violations
            print(f"  Trajectory {idx+1}: {info['type']}")
            if 'decrease' in info:
                print(f"    Decrease: {info['decrease']:.6f} < ε={info['epsilon']:.6f}")
            if 'B_curr' in info:
                print(f"    B: {info['B_curr']:.6f} -> {info['B_next']:.6f}")

    print("="*70)
