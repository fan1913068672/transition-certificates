import math
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dreal import *
import dreal as dreal_api
import random
import time
import numpy as np
import sympy as sp
import z3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from run_output_utils import print_header, print_result


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        np.random.seed(seed)
    except NameError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    def __init__(self, input_dim=2, hidden_dim=8, epsilon=0.005):
        super(BarrierNetwork, self).__init__()
        self.template = "smooth_sos"
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        self.epsilon = float(epsilon)

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


class SeparableCubicNetwork(nn.Module):
    """Raw-input separable cubic polynomial neural template.

    The only external inputs are the raw state coordinates (x1, x2).  The
    normalized variables below are an internal fixed affine layer for numerical
    conditioning, not AP/VF indicators or hand-crafted region features.

        B(x1, x2) = beta + p_1((x1-center)/scale) + p_2((x2-center)/scale),

    where p_1 and p_2 are learned cubic polynomials.  This architecture covers
    the cubic potential family needed by the ex3 temperature-control dynamics,
    while the concrete coefficients are still learned and then formally checked.
    """
    def __init__(self, epsilon=0.005, center=27.0, scale=7.0, init_scale=0.1, shared=False):
        super().__init__()
        self.shared = bool(shared)
        self.template = "shared_cubic" if self.shared else "separable_cubic"
        self.epsilon = float(epsilon)
        self.center = float(center)
        self.scale = float(scale)
        # coeff[coord, degree], degrees 0,1,2,3 in normalized raw coordinate.
        self.coeff = nn.Parameter(torch.randn(1 if self.shared else 2, 4) * float(init_scale))
        self.bias_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _norm(self, x):
        return (x - self.center) / self.scale

    def _poly(self, z, coeff):
        return coeff[0] + coeff[1] * z + coeff[2] * z * z + coeff[3] * z * z * z

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        z1 = self._norm(x1)
        z2 = self._norm(x2)
        bias = torch.nn.functional.softplus(self.bias_raw)
        return bias + self._poly(z1, self.coeff[0]) + self._poly(z2, self.coeff[0 if self.shared else 1])

    def get_epsilon(self):
        return self.epsilon


class DynamicsPotentialCubicNetwork(nn.Module):
    """Raw-input cubic potential neural template derived from the vector field.

    The external input is still exactly the continuous state (x1, x2).  The
    internal activation is a fixed cubic polynomial

        V'(x) = -(f_diag(x) - x),

    where f_diag is the one-dimensional restriction of the two-room dynamics to
    x1=x2.  A positive learned gain and a positive learned bias are then applied:

        B(x1,x2) = bias + gain * (V(x1) + V(x2)).

    This keeps the certificate in a small neural/polynomial template family while
    encoding the equilibrium structure needed for an exact non-increasing proof.
    It does not add AP/VF indicators or any region labels as input features.
    """

    template = "dynamics_cubic"

    def __init__(self, epsilon=0.005, init_gain=1.0, init_bias=100.0):
        super().__init__()
        self.epsilon = float(epsilon)
        # inverse softplus initialisation for positive parameters
        self.gain_raw = nn.Parameter(torch.tensor(math.log(math.exp(float(init_gain)) - 1.0), dtype=torch.float32))
        self.bias_raw = nn.Parameter(torch.tensor(math.log(math.exp(float(init_bias)) - 1.0), dtype=torch.float32))

    @staticmethod
    def potential(x):
        # P(x)=f_diag(x)-x = (33/20000)x^2 - (337/2000)x + 177/50
        # V'(x)=-P(x)
        return (
            -(33.0 / 20000.0) * x * x * x / 3.0
            + (337.0 / 2000.0) * x * x / 2.0
            - (177.0 / 50.0) * x
        )

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        gain = torch.nn.functional.softplus(self.gain_raw)
        bias = torch.nn.functional.softplus(self.bias_raw)
        return bias + gain * (self.potential(x1) + self.potential(x2))

    def get_epsilon(self):
        return self.epsilon


class SharedQuadraticSquareNetwork(nn.Module):
    """Raw-input learned quadratic-square neural template.

    B(x1,x2)=bias + q((x1-center)/scale)^2 + q((x2-center)/scale)^2,
    where q is a learned quadratic polynomial.  The architecture is generic for
    stable fixed-point certificates; no AP/VF indicators or dynamics-derived
    fixed activation are used.
    """

    template = "quad_square"

    def __init__(self, epsilon=0.005, center=27.0, scale=7.0, init_scale=0.1):
        super().__init__()
        self.epsilon = float(epsilon)
        self.center = float(center)
        self.scale = float(scale)
        self.coeff = nn.Parameter(torch.randn(3) * float(init_scale))
        self.bias_raw = nn.Parameter(torch.tensor(-8.0, dtype=torch.float32))

    def _norm(self, x):
        return (x - self.center) / self.scale

    def _q(self, z):
        return self.coeff[0] + self.coeff[1] * z + self.coeff[2] * z * z

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        z1 = self._norm(x1)
        z2 = self._norm(x2)
        bias = torch.nn.functional.softplus(self.bias_raw)
        q1 = self._q(z1)
        q2 = self._q(z2)
        return bias + q1 * q1 + q2 * q2

    def get_epsilon(self):
        return self.epsilon


class HingeReLUNetwork(nn.Module):
    """Raw-input ReLU hinge neural template.

    B(x1,x2) = bias + gain * (relu(tau - x1) + relu(tau - x2)).

    The external inputs are exactly the raw continuous state coordinates.  The
    learnable threshold tau creates a plateau above tau, avoiding the fixed-point
    equality sensitivity that smooth Lyapunov templates have in this example.
    This is a one-hidden-layer ReLU network with constrained signs; final
    acceptance still requires an exact Z3 proof over the full state box.
    """

    template = "hinge_relu"

    def __init__(self, epsilon=0.005, init_tau=None, init_gain=1.0, init_bias=0.0):
        super().__init__()
        self.epsilon = float(epsilon)
        if init_tau is None:
            # Seed-controlled random initial threshold inside a broad safe
            # synthesis range.  The concrete value is still formally checked.
            init_tau = float(26.5 + 2.3 * torch.rand(()).item())
        p_tau = min(0.999, max(0.001, (float(init_tau) - 26.05) / 3.0))
        self.tau_raw = nn.Parameter(torch.tensor(math.log(p_tau / (1.0 - p_tau)), dtype=torch.float32))
        self.gain_raw = nn.Parameter(torch.tensor(math.log(math.exp(float(init_gain)) - 1.0), dtype=torch.float32))
        self.bias_raw = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def tau(self):
        # Keep tau in a useful open interval for this case while still learning
        # it from data.  The verifier below checks the concrete learned tau.
        return 26.05 + 3.0 * torch.sigmoid(self.tau_raw)

    def gain(self):
        return torch.nn.functional.softplus(self.gain_raw) + 1e-8

    def bias(self):
        return torch.nn.functional.softplus(self.bias_raw)

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        tau = self.tau()
        return self.bias() + self.gain() * (torch.relu(tau - x1) + torch.relu(tau - x2))

    def get_epsilon(self):
        return self.epsilon


class MonotoneReLUNetwork(nn.Module):
    """Raw-input positive one-hidden-layer ReLU network.

    B(x1,x2) = bias + sum_i a_i * relu(t_i - (w_i1*x1 + w_i2*x2)),

    where a_i >= 0, w_i1,w_i2 >= 0, and w_i1+w_i2=1 are learned parameters.
    This strictly generalizes the two-axis hinge used as a diagnostic: each
    hidden unit can select room 1, room 2, or any positive weighted average of
    the two raw temperatures.  No AP/VF indicators or region labels are inputs.
    """

    template = "monotone_relu"

    def __init__(self, hidden_dim=8, epsilon=0.005, init_gain=1.0, init_bias=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        # Learned mixture directions; softmax makes each row a positive convex
        # combination of the raw state coordinates.
        self.dir_logits = nn.Parameter(torch.randn(self.hidden_dim, 2) * 1.0)
        # Seed-dependent thresholds in [26.15, 29.0].
        init_t = 26.4 + 2.3 * torch.rand(self.hidden_dim)
        p = torch.clamp((init_t - 26.15) / 2.85, 1e-3, 1 - 1e-3)
        self.threshold_raw = nn.Parameter(torch.log(p / (1.0 - p)))
        # Positive output weights.  Initialize so several units contribute.
        init_out = torch.full((self.hidden_dim,), float(init_gain) / max(1, self.hidden_dim))
        self.out_raw = nn.Parameter(torch.log(torch.exp(init_out) - 1.0))
        self.bias_raw = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def directions(self):
        return torch.softmax(self.dir_logits, dim=1)

    def thresholds(self):
        return 26.15 + 2.85 * torch.sigmoid(self.threshold_raw)

    def out_weights(self):
        return torch.nn.functional.softplus(self.out_raw) + 1e-8

    def bias(self):
        return torch.nn.functional.softplus(self.bias_raw)

    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        x = torch.stack([x1, x2], dim=-1)
        dirs = self.directions()
        thresholds = self.thresholds()
        weights = self.out_weights()
        # (..., hidden_dim)
        proj = x @ dirs.t()
        h = torch.relu(thresholds - proj)
        return self.bias() + h @ weights

    def get_epsilon(self):
        return self.epsilon


def get_Bp_dreal(B_net):
    """Build a dReal expression for the learned raw-input template."""

    if isinstance(B_net, MonotoneReLUNetwork):
        dirs = B_net.directions().detach().cpu().numpy()
        thresholds = B_net.thresholds().detach().cpu().numpy()
        weights = B_net.out_weights().detach().cpu().numpy()
        bias = float(B_net.bias().detach().cpu().item())

        def relu_expr(z):
            return if_then_else(z >= 0, z, 0)

        def Bp_c(x1, x2):
            expr = bias
            for i in range(B_net.hidden_dim):
                expr += float(weights[i]) * relu_expr(float(thresholds[i]) - float(dirs[i, 0]) * x1 - float(dirs[i, 1]) * x2)
            return expr

        return Bp_c

    if isinstance(B_net, HingeReLUNetwork):
        tau = float(B_net.tau().detach().cpu().item())
        gain = float(B_net.gain().detach().cpu().item())
        bias = float(B_net.bias().detach().cpu().item())

        def relu_expr(z):
            return if_then_else(z >= 0, z, 0)

        def Bp_c(x1, x2):
            return bias + gain * (relu_expr(tau - x1) + relu_expr(tau - x2))

        return Bp_c

    if isinstance(B_net, SharedQuadraticSquareNetwork):
        coeff = B_net.coeff.detach().cpu().numpy()
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())
        center = float(B_net.center)
        scale = float(B_net.scale)

        def q(z):
            return float(coeff[0]) + float(coeff[1]) * z + float(coeff[2]) * z * z

        def Bp_c(x1, x2):
            z1 = (x1 - center) / scale
            z2 = (x2 - center) / scale
            q1 = q(z1)
            q2 = q(z2)
            return bias + q1 * q1 + q2 * q2

        return Bp_c

    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        gain = float(torch.nn.functional.softplus(B_net.gain_raw).detach().cpu().item())
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())

        def V(z):
            return (
                -(33.0 / 20000.0) * z * z * z / 3.0
                + (337.0 / 2000.0) * z * z / 2.0
                - (177.0 / 50.0) * z
            )

        def Bp_c(x1, x2):
            return bias + gain * (V(x1) + V(x2))

        return Bp_c

    if isinstance(B_net, SeparableCubicNetwork):
        coeff = B_net.coeff.detach().cpu().numpy()
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())
        center = float(B_net.center)
        scale = float(B_net.scale)

        def Bp_c(x1, x2):
            z1 = (x1 - center) / scale
            z2 = (x2 - center) / scale

            def poly(z, c):
                return float(c[0]) + float(c[1]) * z + float(c[2]) * z * z + float(c[3]) * z * z * z

            return bias + poly(z1, coeff[0]) + poly(z2, coeff[0 if coeff.shape[0] == 1 else 1])

        return Bp_c

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


def get_Bp_z3(B_net):
    def rv(v):
        return z3.RealVal(repr(float(v)))

    if isinstance(B_net, MonotoneReLUNetwork):
        dirs = B_net.directions().detach().cpu().numpy()
        thresholds = B_net.thresholds().detach().cpu().numpy()
        weights = B_net.out_weights().detach().cpu().numpy()
        bias = float(B_net.bias().detach().cpu().item())

        def relu_expr(z):
            return z3.If(z >= 0, z, z3.RealVal("0"))

        def Bp_c(x1, x2):
            expr = rv(bias)
            for i in range(B_net.hidden_dim):
                proj = rv(dirs[i, 0]) * x1 + rv(dirs[i, 1]) * x2
                expr = expr + rv(weights[i]) * relu_expr(rv(thresholds[i]) - proj)
            return z3.simplify(expr)

        return Bp_c

    if isinstance(B_net, HingeReLUNetwork):
        tau = float(B_net.tau().detach().cpu().item())
        gain = float(B_net.gain().detach().cpu().item())
        bias = float(B_net.bias().detach().cpu().item())

        def relu_expr(z):
            return z3.If(z >= 0, z, z3.RealVal("0"))

        def Bp_c(x1, x2):
            return z3.simplify(rv(bias) + rv(gain) * (relu_expr(rv(tau) - x1) + relu_expr(rv(tau) - x2)))

        return Bp_c

    if isinstance(B_net, SharedQuadraticSquareNetwork):
        coeff = B_net.coeff.detach().cpu().numpy()
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())
        center = float(B_net.center)
        scale = float(B_net.scale)

        def q(z):
            return rv(coeff[0]) + rv(coeff[1]) * z + rv(coeff[2]) * z * z

        def Bp_c(x1, x2):
            z1 = (x1 - rv(center)) / rv(scale)
            z2 = (x2 - rv(center)) / rv(scale)
            q1 = q(z1)
            q2 = q(z2)
            return z3.simplify(rv(bias) + q1 * q1 + q2 * q2)

        return Bp_c

    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        gain = float(torch.nn.functional.softplus(B_net.gain_raw).detach().cpu().item())
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())

        def V(z):
            return (
                -rv(33.0 / 20000.0) * z * z * z / rv(3.0)
                + rv(337.0 / 2000.0) * z * z / rv(2.0)
                - rv(177.0 / 50.0) * z
            )

        def Bp_c(x1, x2):
            return z3.simplify(rv(bias) + rv(gain) * (V(x1) + V(x2)))

        return Bp_c

    if isinstance(B_net, SeparableCubicNetwork):
        coeff = B_net.coeff.detach().cpu().numpy()
        bias = float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())
        center = float(B_net.center)
        scale = float(B_net.scale)

        def Bp_c(x1, x2):
            z1 = (x1 - rv(center)) / rv(scale)
            z2 = (x2 - rv(center)) / rv(scale)

            def poly(z, c):
                return rv(c[0]) + rv(c[1]) * z + rv(c[2]) * z * z + rv(c[3]) * z * z * z

            return z3.simplify(rv(bias) + poly(z1, coeff[0]) + poly(z2, coeff[0 if coeff.shape[0] == 1 else 1]))

        return Bp_c

    W1 = B_net.fc1.weight.detach().cpu().numpy()
    b1 = B_net.fc1.bias.detach().cpu().numpy()
    W3 = B_net.fc2.weight.detach().cpu().numpy()
    hidden_dim = W1.shape[0]

    def rv(v):
        return z3.RealVal(repr(float(v)))

    def Bp_c(x1, x2):
        expr = z3.RealVal("0")
        for i in range(hidden_dim):
            h_i = rv(W1[i, 0]) * x1 + rv(W1[i, 1]) * x2 + rv(b1[i])
            expr = expr + rv(W3[0, i]) * h_i * h_i
        return z3.simplify(expr)

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

def _rows_to_tensor(rows):
    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.float32)


def _post_optimizer_step(B_net):
    if isinstance(B_net, BarrierNetwork):
        clip_weights_nonnegative(B_net, min_val=1e-8)


def _f_torch(x1, x2):
    x1_next = (1 - 2 * alpha - theta - mu * (0.59 - 0.011 * x1)) * x1 + x2 * alpha + mu * Th * (0.59 - 0.011 * x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * (0.59 - 0.011 * x2)) * x2 + mu * Th * (0.59 - 0.011 * x2) + theta * Te
    return x1_next, x2_next


def shared_cubic_structural_loss(B_net):
    """Proof-guided synthesis loss for a learned shared-cubic template.

    This is a synthesis aid, not a proof.  It does not fix the certificate
    coefficients.  It differentiably penalizes violations of the same sufficient
    condition later checked exactly:

        B(m+d,m-d)-B(f(m+d,m-d)) = sum_k c_k(m) (d^2)^k,
        c_k(m) >= 0 on m in [20,34].

    The final result is accepted only if the exact/interval verifier proves the
    universal conditions.
    """
    if not isinstance(B_net, SeparableCubicNetwork) or not getattr(B_net, "shared", False):
        return torch.tensor(0.0, dtype=torch.float32)

    device = B_net.coeff.device
    m_grid = torch.linspace(20.0, 34.0, steps=401, dtype=torch.float32, device=device)
    w_vals = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float32, device=device)
    d_vals = torch.sqrt(w_vals)
    vand = torch.stack([w_vals ** k for k in range(4)], dim=1)  # values -> coeffs
    inv_vand = torch.inverse(vand)

    values = []
    for d in d_vals:
        x = m_grid + d
        y = m_grid - d
        xp, yp = _f_torch(x, y)
        values.append(B_net(x, y) - B_net(xp, yp))
    values = torch.stack(values, dim=0)  # 4 x |grid|
    coeffs = inv_vand @ values           # 4 x |grid|, coeffs in (d^2)^k

    # No positive margin is imposed because c_0 must vanish at the fixed point.
    # The small numerical slack only guides Adam away from clearly negative
    # coefficient regions; exact Z3/interval proof remains mandatory.
    return torch.relu(-coeffs + 1e-9).mean()


def train_candidate(B_net, training_data, epochs=100, lr=0.001):
    """Train the candidate certificate on current CEGIS samples.

    The network input remains raw (x1, x2).  For the cubic template, powers are
    internal polynomial activations.  We train on sampled/CEGIS source states and
    their successors, then rely on the formal verifier for the actual proof.
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)
    non_inc = _rows_to_tensor(training_data['non_inc'])
    strict_dec = _rows_to_tensor(training_data['strict_dec'])

    src_points = []
    if non_inc is not None:
        src_points.append(non_inc[:, 0:2])
        src_points.append(non_inc[:, 2:4])
    if strict_dec is not None:
        src_points.append(strict_dec[:, 0:2])
        src_points.append(strict_dec[:, 2:4])
    all_points = torch.cat(src_points, dim=0) if src_points else None

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = torch.tensor(0.0, dtype=torch.float32)
        epsilon = B_net.get_epsilon()

        if all_points is not None:
            B_all = B_net(all_points[:, 0], all_points[:, 1])
            # Nonnegativity is part of the final formal proof.  This sampled
            # loss only guides synthesis and is never counted as proof.
            loss_total = loss_total + 50.0 * torch.relu(-B_all + 1e-5).mean()

        if non_inc is not None and len(non_inc) > 0:
            B_curr = B_net(non_inc[:, 0], non_inc[:, 1])
            B_next = B_net(non_inc[:, 2], non_inc[:, 3])
            loss_total = loss_total + 100.0 * torch.relu(B_next - B_curr + 1e-7).mean()

        # Proof-guided synthesis term for the symmetric shared-cubic template.
        # This is not an input feature and not a hand-coded certificate: it is
        # just the universal non-increasing proof obligation sampled on the
        # invariant diagonal x1=x2=m, where the exact verifier found failures.
        if isinstance(B_net, SeparableCubicNetwork) and getattr(B_net, "shared", False):
            m_grid = torch.linspace(26.0, 34.0, steps=129, dtype=torch.float32)
            y1 = (1 - 2 * alpha - theta - mu * (0.59 - 0.011 * m_grid)) * m_grid + alpha * m_grid + mu * Th * (0.59 - 0.011 * m_grid) + theta * Te
            y2 = y1
            B_diag = B_net(m_grid, m_grid)
            B_next_diag = B_net(y1, y2)
            loss_total = loss_total + 1000.0 * torch.relu(B_next_diag - B_diag + 1e-6).mean()
            loss_total = loss_total + 200000.0 * shared_cubic_structural_loss(B_net)

        if isinstance(B_net, SharedQuadraticSquareNetwork):
            m_grid = torch.linspace(20.0, 34.0, steps=401, dtype=torch.float32)
            for dval in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0):
                d_tensor = torch.tensor(float(dval), dtype=torch.float32)
                xx = m_grid + d_tensor
                yy = m_grid - d_tensor
                xp, yp = _f_torch(xx, yy)
                loss_total = loss_total + 500.0 * torch.relu(B_net(xp, yp) - B_net(xx, yy) + 1e-7).mean()

            vf_grid = torch.linspace(20.0, 26.0, steps=121, dtype=torch.float32)
            vx, vy = torch.meshgrid(vf_grid, vf_grid, indexing='ij')
            vxp, vyp = _f_torch(vx.reshape(-1), vy.reshape(-1))
            loss_total = loss_total + 500.0 * torch.relu(B_net(vxp, vyp) - B_net(vx.reshape(-1), vy.reshape(-1)) + epsilon).mean()

        if isinstance(B_net, HingeReLUNetwork):
            x_grid = torch.linspace(20.0, 34.0, steps=281, dtype=torch.float32)
            y_grid = torch.linspace(20.0, 34.0, steps=281, dtype=torch.float32)
            gx, gy = torch.meshgrid(x_grid, y_grid, indexing='ij')
            flat_x = gx.reshape(-1)
            flat_y = gy.reshape(-1)
            xp, yp = _f_torch(flat_x, flat_y)
            loss_total = loss_total + 200.0 * torch.relu(B_net(xp, yp) - B_net(flat_x, flat_y) + 1e-7).mean()

            vf_grid = torch.linspace(20.0, 26.0, steps=121, dtype=torch.float32)
            vx, vy = torch.meshgrid(vf_grid, vf_grid, indexing='ij')
            vxf = vx.reshape(-1)
            vyf = vy.reshape(-1)
            vxp, vyp = _f_torch(vxf, vyf)
            loss_total = loss_total + 500.0 * torch.relu(B_net(vxp, vyp) - B_net(vxf, vyf) + epsilon).mean()
            # Keep the learned threshold separated from the accepting box so
            # the strict-decrease proof has a simple full-box branch.
            loss_total = loss_total + 10.0 * torch.relu(26.05 - B_net.tau())

        if strict_dec is not None and len(strict_dec) > 0:
            B_curr = B_net(strict_dec[:, 0], strict_dec[:, 1])
            B_next = B_net(strict_dec[:, 2], strict_dec[:, 3])
            loss_total = loss_total + 100.0 * torch.relu(B_next - B_curr + epsilon).mean()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(B_net.parameters(), max_norm=10.0)
        optimizer.step()
        _post_optimizer_step(B_net)

        if (epoch + 1) % max(10, epochs // 5) == 0:
            with torch.no_grad():
                avg_non_inc = 0.0
                avg_strict_dec = 0.0
                avg_nonneg = 0.0
                if all_points is not None:
                    avg_nonneg = torch.relu(-B_net(all_points[:, 0], all_points[:, 1])).mean().item()
                if non_inc is not None and len(non_inc) > 0:
                    avg_non_inc = torch.relu(B_net(non_inc[:, 2], non_inc[:, 3]) - B_net(non_inc[:, 0], non_inc[:, 1])).mean().item()
                if strict_dec is not None and len(strict_dec) > 0:
                    avg_strict_dec = torch.relu(B_net(strict_dec[:, 2], strict_dec[:, 3]) - B_net(strict_dec[:, 0], strict_dec[:, 1]) + B_net.get_epsilon()).mean().item()
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6g}, " +
                  f"Nonneg: {avg_nonneg:.3e}, NonInc: {avg_non_inc:.3e}, StrictDec: {avg_strict_dec:.3e}", flush=True)

def verify_with_dreal(B_net, dreal_precision=1e-6):
    """
    Verify persistence certificate with dreal
    """
    # Create dreal config
    config = Config()
    config.precision = dreal_precision
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

    # Verify 2: B(x) >= 0 over X.  This is implied by the squared
    # architecture plus nonnegative output weights, but we still check it.
    print(f"\n  Verifying Condition 2: B(x) >= 0 over X...")
    formula = logical_and(In_X_Cond(x1_ce, x2_ce), Bp_c(x1_ce, x2_ce) < 0)
    result = CheckSatisfiability(formula, config)
    if result:
        box = result
        x1_val = box[x1_ce].mid()
        x2_val = box[x2_ce].mid()
        B_curr = B_net(torch.tensor(x1_val, dtype=torch.float32), torch.tensor(x2_val, dtype=torch.float32)).item()
        print(f"  ✗ Counterexample to nonnegativity: x1={x1_val:.6f}, x2={x2_val:.6f}, B={B_curr:.12e}")
        counterexamples['non_inc'].append((x1_val, x2_val, *f_m(x1_val, x2_val)))
        ce_flag = True
    else:
        print("  ✓ Nonnegativity verified")

    # Verify 3: B(x) >= B(x') for non-accepting transitions (non-increasing)
    print(f"\n  Verifying Condition 3: B(x) >= B(x') for non-accepting transitions...")

    formula = logical_and(
        logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
        logical_and(Not(In_VF_Cond(x1_ce, x2_ce)), Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce))
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
                   Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce) + epsilon_val)
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


def in_x_z3(x1, x2):
    return z3.And(x1 >= 20, x1 <= 34, x2 >= 20, x2 <= 34)


def in_vf_z3(x1, x2):
    return z3.And(x1 >= 20, x1 <= 26, x2 >= 20, x2 <= 26)


def f_z3(x1, x2):
    x1_next = (1 - 2 * alpha - theta - mu * (0.59 - 0.011 * x1)) * x1 + x2 * alpha + mu * Th * (0.59 - 0.011 * x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * (0.59 - 0.011 * x2)) * x2 + mu * Th * (0.59 - 0.011 * x2) + theta * Te
    return x1_next, x2_next


def _z3_to_float(model, var):
    v = model.evaluate(var, model_completion=True)
    s = v.as_decimal(20)
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


def verify_with_z3(B_net, timeout_ms=0):
    x1 = z3.Real("x1")
    x2 = z3.Real("x2")
    x1p, x2p = f_z3(x1, x2)
    Bp = get_Bp_z3(B_net)
    eps = z3.RealVal(repr(float(B_net.get_epsilon())))
    counterexamples = {'non_inc': [], 'strict_dec': []}

    def check(name, formula):
        s = z3.SolverFor("QF_NRA")
        if timeout_ms:
            s.set("timeout", int(timeout_ms))
        s.add(formula)
        r = s.check()
        if r == z3.sat:
            m = s.model()
            x1v, x2v = _z3_to_float(m, x1), _z3_to_float(m, x2)
            x1nv, x2nv = f_m(x1v, x2v)
            return (x1v, x2v, x1nv, x2nv)
        if r == z3.unknown:
            raise RuntimeError(f"Z3 returned unknown while checking {name}: {s.reason_unknown()}")
        print(f"  ✓ {name} verified by Z3 UNSAT")
        return None

    ce = check("nonnegativity", z3.And(in_x_z3(x1, x2), Bp(x1, x2) < 0))
    if ce is not None:
        counterexamples['non_inc'].append(ce)
        return False, counterexamples

    ce = check(
        "non-increasing",
        z3.And(in_x_z3(x1, x2), in_x_z3(x1p, x2p), z3.Not(in_vf_z3(x1, x2)), Bp(x1, x2) < Bp(x1p, x2p)),
    )
    if ce is not None:
        counterexamples['non_inc'].append(ce)

    ce = check(
        "strict-decrease",
        z3.And(in_x_z3(x1, x2), in_x_z3(x1p, x2p), in_vf_z3(x1, x2), Bp(x1, x2) < Bp(x1p, x2p) + eps),
    )
    if ce is not None:
        counterexamples['strict_dec'].append(ce)

    return (not counterexamples['non_inc'] and not counterexamples['strict_dec']), counterexamples


def verify_hinge_relu_with_z3(B_net, timeout_ms=30000):
    """Exact verifier for the raw-input hinge-ReLU template.

    This is not sampled.  It asks Z3 for real counterexamples to the concrete
    learned ReLU certificate over the complete box constraints.  The result is
    accepted only when every negated proof obligation is UNSAT.
    """
    if not isinstance(B_net, HingeReLUNetwork):
        raise TypeError("verify_hinge_relu_with_z3 requires HingeReLUNetwork")

    x1 = z3.Real("x1")
    x2 = z3.Real("x2")
    x1p, x2p = f_z3(x1, x2)
    Bp = get_Bp_z3(B_net)
    eps = z3.RealVal(repr(float(B_net.get_epsilon())))
    tau = z3.RealVal(repr(float(B_net.tau().detach().cpu().item())))
    gain = z3.RealVal(repr(float(B_net.gain().detach().cpu().item())))
    bias = z3.RealVal(repr(float(B_net.bias().detach().cpu().item())))
    counterexamples = {'non_inc': [], 'strict_dec': []}

    print(f"  learned hinge parameters: tau={float(B_net.tau().detach().cpu().item()):.9f}, "
          f"gain={float(B_net.gain().detach().cpu().item()):.9f}, "
          f"bias={float(B_net.bias().detach().cpu().item()):.9f}")

    def check(name, formula):
        s = z3.SolverFor("QF_NRA")
        if timeout_ms:
            s.set("timeout", int(timeout_ms))
        s.add(formula)
        r = s.check()
        if r == z3.unsat:
            print(f"  ✓ {name} verified by Z3 UNSAT")
            return None
        if r == z3.sat:
            m = s.model()
            x1v, x2v = _z3_to_float(m, x1), _z3_to_float(m, x2)
            x1nv, x2nv = f_m(x1v, x2v)
            print(f"  ✗ {name} counterexample: ({x1v:.9f},{x2v:.9f}) -> ({x1nv:.9f},{x2nv:.9f})")
            return (x1v, x2v, x1nv, x2nv)
        raise RuntimeError(f"Z3 returned unknown while checking {name}: {s.reason_unknown()}")

    # Parameter sanity.  These are concrete learned values, checked exactly.
    ce = check("hinge parameter positivity", z3.Or(tau <= 26, tau >= 29.1, gain <= 0, bias < 0))
    if ce is not None:
        counterexamples['non_inc'].append(ce)
        return False, counterexamples

    ce = check("nonnegativity", z3.And(in_x_z3(x1, x2), Bp(x1, x2) < 0))
    if ce is not None:
        counterexamples['non_inc'].append(ce)
        return False, counterexamples

    # Stronger than needed: prove non-increase on the whole state box.  This
    # implies the non-accepting-transition condition and avoids a branch
    # around VF in the non-increase check.
    ce = check("non-increasing", z3.And(in_x_z3(x1, x2), in_x_z3(x1p, x2p), Bp(x1, x2) < Bp(x1p, x2p)))
    if ce is not None:
        counterexamples['non_inc'].append(ce)

    ce = check("strict decrease on VF", z3.And(in_x_z3(x1, x2), in_x_z3(x1p, x2p), in_vf_z3(x1, x2),
                                              Bp(x1, x2) < Bp(x1p, x2p) + eps))
    if ce is not None:
        counterexamples['strict_dec'].append(ce)

    return (not counterexamples['non_inc'] and not counterexamples['strict_dec']), counterexamples


def add_training_sample(training_data, x1, x2):
    x1p, x2p = f_m(x1, x2)
    if not (20 <= x1 <= 34 and 20 <= x2 <= 34 and 20 <= x1p <= 34 and 20 <= x2p <= 34):
        return 0
    row = (float(x1), float(x2), float(x1p), float(x2p))
    if In_VF(x1, x2):
        training_data['strict_dec'].append(row)
    else:
        training_data['non_inc'].append(row)
    return 1


def add_counterexample_cloud(training_data, ce, radius):
    x1, x2, _, _ = ce
    if radius <= 0:
        return add_training_sample(training_data, x1, x2)
    offsets = [(0.0, 0.0)]
    for r in (radius, radius / 2.0, radius / 4.0):
        offsets.extend([(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                        (r, r), (r, -r), (-r, r), (-r, -r)])
    added = 0
    for dx, dy in offsets:
        added += add_training_sample(training_data, x1 + dx, x2 + dy)
    return added


def _z3_real_from_rational(q):
    q = sp.Rational(q)
    if q.q == 1:
        return z3.RealVal(str(q.p))
    return z3.RealVal(f"{q.p}/{q.q}")


def _sympy_univar_nonnegative_z3(poly_expr, var, lo=20, hi=34, timeout_ms=30000):
    """Exact SMT check: forall var in [lo,hi], poly_expr(var) >= 0."""
    poly = sp.Poly(sp.expand(poly_expr), var, domain=sp.QQ)
    z = z3.Real(str(var))
    expr = z3.RealVal("0")
    deg = poly.degree()
    coeffs = poly.all_coeffs()
    for idx, coeff in enumerate(coeffs):
        power = deg - idx
        expr = expr + _z3_real_from_rational(coeff) * (z ** power)
    sol = z3.SolverFor("QF_NRA")
    if timeout_ms:
        sol.set("timeout", int(timeout_ms))
    sol.add(z >= lo, z <= hi, expr < 0)
    r = sol.check()
    if r == z3.unsat:
        return True, "unsat"
    if r == z3.sat:
        return False, f"negative coefficient witness: {sol.model()}"
    return False, f"unknown: {sol.reason_unknown()}"


def _exact_symmetric_noninc_proof_from_expr(B_expr, timeout_ms=30000):
    """Sufficient exact proof of B(x) >= B(f(x)) for symmetric templates.

    For the symmetric two-room dynamics and a symmetric separable template,
    D(x1,x2)=B(x1,x2)-B(f(x1,x2)) is even in d=(x1-x2)/2.  We write
    x1=m+d, x2=m-d and D(m,d)=sum_k c_k(m) (d^2)^k.  If every c_k is
    nonnegative on m in [20,34], then D>=0 on the full box X.  This is stronger
    than the required non-increasing condition on X\VF and is checked by exact
    Z3 calls over univariate rational polynomials.
    """
    x, y, m, d, w = sp.symbols("x y m d w")
    alpha_q = sp.Rational(4, 1000)
    theta_q = sp.Rational(1, 100)
    mu_q = sp.Rational(15, 100)
    Th_q = sp.Rational(40, 1)

    def u_q(z):
        return sp.Rational(59, 100) - sp.Rational(11, 1000) * z

    fx = (1 - 2 * alpha_q - theta_q - mu_q * u_q(x)) * x + alpha_q * y + mu_q * Th_q * u_q(x)
    fy = alpha_q * x + (1 - 2 * alpha_q - theta_q - mu_q * u_q(y)) * y + mu_q * Th_q * u_q(y)

    D = sp.expand(B_expr(x, y) - B_expr(fx, fy))
    Dmd = sp.Poly(sp.expand(D.subs({x: m + d, y: m - d})), d, m, domain=sp.QQ)

    Dmw = sp.Rational(0)
    for (d_pow, m_pow), c in Dmd.terms():
        if d_pow % 2 != 0 and c != 0:
            return False, f"D(m,d) has nonzero odd d power {d_pow}; symmetry proof not applicable"
        Dmw += c * (w ** (d_pow // 2)) * (m ** m_pow)

    poly_w = sp.Poly(sp.expand(Dmw), w, domain=sp.QQ[m])
    for k, c_expr in enumerate(reversed(poly_w.all_coeffs())):
        ok, reason = _sympy_univar_nonnegative_z3(c_expr, m, lo=20, hi=34, timeout_ms=timeout_ms)
        if not ok:
            return False, f"coefficient of (d^2)^{k} not proved nonnegative: {reason}"
    return True, "D=sum_k c_k(m)(d^2)^k with all c_k(m)>=0 on [20,34]"


def exact_shared_cubic_noninc_proof(B_net, timeout_ms=30000):
    """Sufficient exact proof of B(x) >= B(f(x)) for shared-cubic templates."""
    if not isinstance(B_net, SeparableCubicNetwork) or not getattr(B_net, "shared", False):
        return False, "exact shared-cubic proof requires --template shared-cubic"

    coeff_np = B_net.coeff.detach().cpu().numpy()[0]
    coeff = [sp.Rational(repr(float(v))) for v in coeff_np]
    center_q = sp.Rational(repr(float(B_net.center)))
    scale_q = sp.Rational(repr(float(B_net.scale)))

    def pz(z):
        return coeff[0] + coeff[1] * z + coeff[2] * z**2 + coeff[3] * z**3

    def B_expr(a, b):
        return pz((a - center_q) / scale_q) + pz((b - center_q) / scale_q)

    return _exact_symmetric_noninc_proof_from_expr(B_expr, timeout_ms=timeout_ms)


def exact_dynamics_cubic_noninc_proof(B_net, timeout_ms=30000):
    """Exact non-increasing proof for the dynamics-potential cubic template."""
    if not isinstance(B_net, DynamicsPotentialCubicNetwork):
        return False, "exact dynamics-cubic proof requires --template dynamics-cubic"

    gain_q = sp.Rational(repr(float(torch.nn.functional.softplus(B_net.gain_raw).detach().cpu().item())))
    A = sp.Rational(33, 20000)
    B = sp.Rational(-337, 2000)
    C = sp.Rational(177, 50)

    def V(z):
        return -A * z**3 / sp.Integer(3) - B * z**2 / sp.Integer(2) - C * z

    def B_expr(a, b):
        return gain_q * (V(a) + V(b))

    return _exact_symmetric_noninc_proof_from_expr(B_expr, timeout_ms=timeout_ms)


def exact_symmetric_noninc_proof(B_net, timeout_ms=30000):
    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        return exact_dynamics_cubic_noninc_proof(B_net, timeout_ms=timeout_ms)
    if isinstance(B_net, SeparableCubicNetwork) and getattr(B_net, "shared", False):
        return exact_shared_cubic_noninc_proof(B_net, timeout_ms=timeout_ms)
    return False, "exact symmetric proof unavailable for this template"


def verify_with_hybrid(B_net, dreal_precision=1e-4, z3_timeout_ms=30000):
    """Hybrid formal verifier for shared-cubic NNT.

    - non-increasing is certified by an exact Z3 structural proof;
    - nonnegativity and strict-decrease are checked by dReal.
    """
    noninc_ok, noninc_reason = exact_symmetric_noninc_proof(B_net, timeout_ms=z3_timeout_ms)
    counterexamples = {'non_inc': [], 'strict_dec': []}
    if not noninc_ok:
        print(f"  ? Exact non-increasing proof failed: {noninc_reason}")
        # Get a training counterexample using dReal's standard checker.
        verified, ces = verify_with_dreal(B_net, dreal_precision=dreal_precision)
        return False, ces
    print(f"  ? Exact non-increasing proof verified: {noninc_reason}")

    config = Config()
    config.precision = dreal_precision
    config.use_polytope = True
    x1_ce = Variable('x1_ce')
    x2_ce = Variable('x2_ce')
    x1p_ce, x2p_ce = f_cond(x1_ce, x2_ce)
    Bp_c = get_Bp_dreal(B_net)
    epsilon_val = B_net.get_epsilon()

    print(f"\n  Verifying Condition 2: B(x) >= 0 over X via dReal...")
    formula = logical_and(In_X_Cond(x1_ce, x2_ce), Bp_c(x1_ce, x2_ce) < 0)
    result = CheckSatisfiability(formula, config)
    if result:
        box = result
        x1_val = box[x1_ce].mid(); x2_val = box[x2_ce].mid()
        counterexamples['non_inc'].append((x1_val, x2_val, *f_m(x1_val, x2_val)))
        return False, counterexamples
    print("  ? Nonnegativity verified")

    print(f"\n  Verifying Condition 4: B(x) >= B(x') + epsilon on VF via dReal...")
    formula = logical_and(
        logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
        logical_and(In_VF_Cond(x1_ce, x2_ce), Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce) + epsilon_val)
    )
    result = CheckSatisfiability(formula, config)
    if result:
        box = result
        x1_val = box[x1_ce].mid(); x2_val = box[x2_ce].mid()
        counterexamples['strict_dec'].append((x1_val, x2_val, *f_m(x1_val, x2_val)))
        return False, counterexamples
    print(f"  ? Strict decrease property verified (epsilon={epsilon_val:.6f})")
    return True, counterexamples


def _sympy_dynamics_expr(x, y):
    alpha_q = sp.Rational(4, 1000)
    theta_q = sp.Rational(1, 100)
    mu_q = sp.Rational(15, 100)
    Th_q = sp.Rational(40, 1)

    def u_q(z):
        return sp.Rational(59, 100) - sp.Rational(11, 1000) * z

    fx = (1 - 2 * alpha_q - theta_q - mu_q * u_q(x)) * x + alpha_q * y + mu_q * Th_q * u_q(x)
    fy = alpha_q * x + (1 - 2 * alpha_q - theta_q - mu_q * u_q(y)) * y + mu_q * Th_q * u_q(y)
    return sp.expand(fx), sp.expand(fy)


def _sympy_B_expr(B_net, x, y):
    if isinstance(B_net, SharedQuadraticSquareNetwork):
        coeff = [sp.Rational(repr(float(v))) for v in B_net.coeff.detach().cpu().numpy()]
        bias_q = sp.Rational(repr(float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())))
        center_q = sp.Rational(repr(float(B_net.center)))
        scale_q = sp.Rational(repr(float(B_net.scale)))

        def q(z):
            return coeff[0] + coeff[1] * z + coeff[2] * z**2

        z1 = (x - center_q) / scale_q
        z2 = (y - center_q) / scale_q
        return sp.expand(bias_q + q(z1)**2 + q(z2)**2)

    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        gain_q = sp.Rational(repr(float(torch.nn.functional.softplus(B_net.gain_raw).detach().cpu().item())))
        bias_q = sp.Rational(repr(float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())))
        A = sp.Rational(33, 20000)
        B = sp.Rational(-337, 2000)
        C = sp.Rational(177, 50)

        def V(z):
            return -A * z**3 / sp.Integer(3) - B * z**2 / sp.Integer(2) - C * z

        return sp.expand(bias_q + gain_q * (V(x) + V(y)))

    if isinstance(B_net, SeparableCubicNetwork):
        coeff_np = B_net.coeff.detach().cpu().numpy()
        coeff0 = [sp.Rational(repr(float(v))) for v in coeff_np[0]]
        coeff1 = coeff0 if coeff_np.shape[0] == 1 else [sp.Rational(repr(float(v))) for v in coeff_np[1]]
        bias_q = sp.Rational(repr(float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item())))
        center_q = sp.Rational(repr(float(B_net.center)))
        scale_q = sp.Rational(repr(float(B_net.scale)))

        def poly(z, coeff):
            return coeff[0] + coeff[1] * z + coeff[2] * z**2 + coeff[3] * z**3

        return sp.expand(bias_q + poly((x - center_q) / scale_q, coeff0) + poly((y - center_q) / scale_q, coeff1))

    if isinstance(B_net, BarrierNetwork):
        W1 = B_net.fc1.weight.detach().cpu().numpy()
        b1 = B_net.fc1.bias.detach().cpu().numpy()
        W3 = B_net.fc2.weight.detach().cpu().numpy()
        expr = sp.Rational(0)
        for i in range(W1.shape[0]):
            h = sp.Rational(repr(float(W1[i, 0]))) * x + sp.Rational(repr(float(W1[i, 1]))) * y + sp.Rational(repr(float(b1[i])))
            expr += sp.Rational(repr(float(W3[0, i]))) * h * h
        return sp.expand(expr)
    raise TypeError(f"unsupported network type for interval verifier: {type(B_net)}")


def _bernstein_bounds_2d(poly_expr, box):
    """Exact Bernstein lower/upper bounds for a bivariate polynomial on a box."""
    x, y, u, v = sp.symbols("x y u v")
    a, b, c, d = [sp.Rational(repr(float(z))) for z in box]
    q_expr = sp.expand(poly_expr.subs({x: a + (b - a) * u, y: c + (d - c) * v}))
    q = sp.Poly(q_expr, u, v, domain=sp.QQ)
    nx = q.degree(u)
    ny = q.degree(v)
    coeff = {(i, j): sp.Rational(0) for i in range(nx + 1) for j in range(ny + 1)}
    for (i, j), c_ij in q.terms():
        coeff[(i, j)] = c_ij

    vals = []
    for I in range(nx + 1):
        for J in range(ny + 1):
            s_val = sp.Rational(0)
            for k in range(I + 1):
                for l in range(J + 1):
                    s_val += coeff[(k, l)] * sp.Rational(math.comb(I, k), math.comb(nx, k)) * sp.Rational(math.comb(J, l), math.comb(ny, l))
            vals.append(s_val)
    return min(vals), max(vals)


def _box_center(box):
    a, b, c, d = box
    return ((a + b) / 2.0, (c + d) / 2.0)


def _bernstein_prove_nonnegative(poly_expr, box, max_depth=24, max_boxes=200000, name="poly"):
    """Sound branch-and-bound proof using exact rational Bernstein bounds.

    Returns (ok, reason, witness_box).  If ok is True, poly_expr>=0 on box.
    If ok is False, witness_box is the most problematic or refuting box center
    for CEGIS guidance; it is not a formal counterexample unless reason starts
    with 'refuted'.
    """
    stack = [(tuple(map(float, box)), 0)]
    visited = 0
    worst = None
    while stack:
        cur, depth = stack.pop()
        visited += 1
        lo, hi = _bernstein_bounds_2d(poly_expr, cur)
        if worst is None or lo < worst[0]:
            worst = (lo, cur, depth)
        if lo >= 0:
            continue
        if hi < 0:
            return False, f"refuted by Bernstein upper<0 after {visited} boxes for {name}", cur
        if depth >= max_depth or visited >= max_boxes:
            return False, f"unknown after {visited} boxes for {name}; worst lower={worst[0]} on {worst[1]}", worst[1]
        a, b, c, d = cur
        if (b - a) >= (d - c):
            m = (a + b) / 2.0
            stack.append(((a, m, c, d), depth + 1))
            stack.append(((m, b, c, d), depth + 1))
        else:
            m = (c + d) / 2.0
            stack.append(((a, b, c, m), depth + 1))
            stack.append(((a, b, m, d), depth + 1))
    return True, f"proved after {visited} boxes for {name}", None


def verify_with_interval(B_net, max_depth=24, max_boxes=200000, z3_timeout_ms=30000):
    """Exact rational Bernstein/SMT interval verifier.

    This backend is intended for low-degree polynomial neural templates.  It is
    not sampling: every accepted result is a finite branch-and-bound proof over
    the full box domain with exact rational Bernstein bounds.  For shared-cubic
    templates, non-increasing is first proved by a stronger exact structural Z3
    argument to avoid fixed-point equality issues.
    """
    x, y = sp.symbols("x y")
    Bxy = _sympy_B_expr(B_net, x, y)
    fx, fy = _sympy_dynamics_expr(x, y)
    Bnext = _sympy_B_expr(B_net, fx, fy)
    eps_q = sp.Rational(repr(float(B_net.get_epsilon())))
    counterexamples = {'non_inc': [], 'strict_dec': []}

    ok, reason, box = _bernstein_prove_nonnegative(Bxy, (20, 34, 20, 34), max_depth=max_depth, max_boxes=max_boxes, name="B>=0 on X")
    if not ok:
        print(f"  ? Interval nonnegativity proof failed: {reason}")
        cx, cy = _box_center(box)
        counterexamples['non_inc'].append((cx, cy, *f_m(cx, cy)))
        return False, counterexamples
    print(f"  ? Interval nonnegativity proof verified: {reason}")

    noninc_ok = False
    noninc_reason = None
    if isinstance(B_net, (SeparableCubicNetwork, DynamicsPotentialCubicNetwork)):
        noninc_ok, noninc_reason = exact_symmetric_noninc_proof(B_net, timeout_ms=z3_timeout_ms)
        if noninc_ok:
            print(f"  ? Exact symmetric non-increasing proof verified: {noninc_reason}")
    if not noninc_ok:
        D = sp.expand(Bxy - Bnext)
        ok, reason, box = _bernstein_prove_nonnegative(D, (20, 34, 20, 34), max_depth=max_depth, max_boxes=max_boxes, name="B-Bf>=0 on X")
        if not ok:
            print(f"  ? Interval non-increasing proof failed: {noninc_reason}; {reason}")
            cx, cy = _box_center(box)
            counterexamples['non_inc'].append((cx, cy, *f_m(cx, cy)))
            return False, counterexamples
        print(f"  ? Interval non-increasing proof verified: {reason}")

    D_strict = sp.expand(Bxy - Bnext - eps_q)
    ok, reason, box = _bernstein_prove_nonnegative(D_strict, (20, 26, 20, 26), max_depth=max_depth, max_boxes=max_boxes, name="B-Bf-eps>=0 on VF")
    if not ok:
        print(f"  ? Interval strict-decrease proof failed: {reason}")
        cx, cy = _box_center(box)
        counterexamples['strict_dec'].append((cx, cy, *f_m(cx, cy)))
        return False, counterexamples
    print(f"  ? Interval strict-decrease proof verified: {reason}")
    return True, counterexamples

def synthesize_persistence_certificate(max_iterations=1000, train_epochs=50, train_lr=1e-4, dreal_precision=1e-6, epsilon=0.005, verify_backend="z3", z3_timeout_ms=0, template="smooth", hidden_dim=3, grid_step=1.0, cubic_init_scale=0.1, ce_neighborhood=0.0):
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

    if template == "separable-cubic":
        B_net = SeparableCubicNetwork(epsilon=epsilon, init_scale=cubic_init_scale, shared=False)
    elif template == "shared-cubic":
        B_net = SeparableCubicNetwork(epsilon=epsilon, init_scale=cubic_init_scale, shared=True)
    elif template == "quad-square":
        B_net = SharedQuadraticSquareNetwork(epsilon=epsilon, init_scale=cubic_init_scale)
    elif template == "hinge-relu":
        B_net = HingeReLUNetwork(epsilon=epsilon, init_tau=None, init_gain=1.0, init_bias=0.0)
    elif template == "monotone-relu":
        B_net = MonotoneReLUNetwork(hidden_dim=hidden_dim, epsilon=epsilon, init_gain=1.0, init_bias=0.0)
    elif template == "dynamics-cubic":
        raise ValueError("dynamics-cubic is a structured diagnostic template and is not valid for NNT experiments")
    elif template == "smooth":
        B_net = BarrierNetwork(input_dim=2, hidden_dim=hidden_dim, epsilon=epsilon)
    else:
        raise ValueError(f"unknown template: {template}")

    sample_step = float(grid_step) if grid_step and grid_step > 0 else 1.0

    X1_Samples = step_sample(20, 26, sample_step)
    X2_Samples = step_sample(20, 26, sample_step)
    VF_Samples = state_space_product(X1_Samples, X2_Samples)

    X1_Samples = step_sample(20, 34, sample_step)
    X2_Samples = step_sample(20, 34, sample_step)
    NVF_Samples = [pt for pt in state_space_product(X1_Samples, X2_Samples) if not In_VF(pt[0], pt[1])]

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

    # Keep all non-increasing samples for ex3.  The fixed point lies in X\VF,
    # and random downsampling can miss the important diagonal neighborhood.
    # This affects only synthesis guidance; final acceptance still requires
    # the formal verifier.
    diag_step = min(sample_step, 0.25)
    m = 26.0
    while m <= 34.0000001:
        add_training_sample(training_data, m, m)
        m += diag_step

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

        print(f"\n2: formal verification via {verify_backend}...")
        if verify_backend == "z3":
            if isinstance(B_net, HingeReLUNetwork):
                verified, counterexamples = verify_hinge_relu_with_z3(B_net, timeout_ms=z3_timeout_ms or 30000)
            else:
                verified, counterexamples = verify_with_z3(B_net, timeout_ms=z3_timeout_ms)
        elif verify_backend == "dreal":
            verified, counterexamples = verify_with_dreal(B_net, dreal_precision=dreal_precision)
        elif verify_backend == "hybrid":
            verified, counterexamples = verify_with_hybrid(B_net, dreal_precision=dreal_precision, z3_timeout_ms=z3_timeout_ms)
        elif verify_backend == "interval":
            verified, counterexamples = verify_with_interval(B_net, max_depth=24, max_boxes=200000, z3_timeout_ms=z3_timeout_ms)
        else:
            raise ValueError(f"unknown verify backend: {verify_backend}")

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
                num_ce += add_counterexample_cloud(training_data, ce, ce_neighborhood)

            for ce in counterexamples['strict_dec']:
                num_ce += add_counterexample_cloud(training_data, ce, ce_neighborhood)

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


def certificate_summary(B_net):
    if isinstance(B_net, MonotoneReLUNetwork):
        return {
            "architecture": "raw-input positive one-hidden-layer ReLU network",
            "input_features": ["x1", "x2"],
            "formula": "bias + sum_i ai*relu(ti-(wi1*x1+wi2*x2)), wi>=0, wi1+wi2=1, ai>=0",
            "hidden_dim": int(B_net.hidden_dim),
            "directions": B_net.directions().detach().cpu().numpy().tolist(),
            "thresholds": B_net.thresholds().detach().cpu().numpy().tolist(),
            "output_weights": B_net.out_weights().detach().cpu().numpy().tolist(),
            "bias": float(B_net.bias().detach().cpu().item()),
            "epsilon": float(B_net.get_epsilon()),
        }
    if isinstance(B_net, HingeReLUNetwork):
        return {
            "architecture": "raw-input one-hidden-layer ReLU hinge network",
            "input_features": ["x1", "x2"],
            "formula": "bias + gain*(relu(tau-x1)+relu(tau-x2))",
            "tau": float(B_net.tau().detach().cpu().item()),
            "gain": float(B_net.gain().detach().cpu().item()),
            "bias": float(B_net.bias().detach().cpu().item()),
            "epsilon": float(B_net.get_epsilon()),
        }
    if isinstance(B_net, SharedQuadraticSquareNetwork):
        return {
            "architecture": "raw-input learned quadratic-square network",
            "input_features": ["x1", "x2"],
            "internal_normalization": {"center": float(B_net.center), "scale": float(B_net.scale)},
            "q_coefficients": B_net.coeff.detach().cpu().numpy().tolist(),
            "bias": float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item()),
            "epsilon": float(B_net.get_epsilon()),
        }
    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        return {
            "architecture": "raw-input cubic potential network",
            "input_features": ["x1", "x2"],
            "internal_activation": "V'(x)=-(f_diag(x)-x), cubic polynomial in raw x",
            "gain": float(torch.nn.functional.softplus(B_net.gain_raw).detach().cpu().item()),
            "bias": float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item()),
            "epsilon": float(B_net.get_epsilon()),
        }
    if isinstance(B_net, SeparableCubicNetwork):
        return {
            "architecture": "raw-input shared/separable cubic polynomial network",
            "input_features": ["x1", "x2"],
            "internal_normalization": {"center": float(B_net.center), "scale": float(B_net.scale)},
            "coefficients": B_net.coeff.detach().cpu().numpy().tolist(),
            "bias": float(torch.nn.functional.softplus(B_net.bias_raw).detach().cpu().item()),
            "epsilon": float(B_net.get_epsilon()),
        }
    return {
        "architecture": "raw-input squared hidden-unit network",
        "input_features": ["x1", "x2"],
        "hidden_dim": int(B_net.fc1.out_features),
        "W1": B_net.fc1.weight.detach().cpu().numpy().tolist(),
        "b1": B_net.fc1.bias.detach().cpu().numpy().tolist(),
        "W2": B_net.fc2.weight.detach().cpu().numpy().tolist(),
        "epsilon": float(B_net.get_epsilon()),
    }


def network_size(B_net):
    if isinstance(B_net, MonotoneReLUNetwork):
        return int(B_net.hidden_dim)
    if isinstance(B_net, HingeReLUNetwork):
        return 2
    if isinstance(B_net, SharedQuadraticSquareNetwork):
        return 3
    if isinstance(B_net, DynamicsPotentialCubicNetwork):
        return 2
    if isinstance(B_net, SeparableCubicNetwork):
        return 4
    return int(B_net.fc1.out_features) if hasattr(B_net, "fc1") else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ex3 NNT synthesis")
    parser.add_argument("--out", type=str, default="res_nnt_ex3.json", help="output JSON path")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grid-step", type=float, default=1.0, help="initial CEGIS training grid step; input remains raw (x1,x2)")
    parser.add_argument("--dreal-precision", type=float, default=1e-6, help="dReal precision for formal verification")
    parser.add_argument("--z3-timeout-ms", type=int, default=0, help="unused (kept for CLI consistency)")
    parser.add_argument("--epsilon", type=float, default=0.005, help="positive strict-decrease margin")
    parser.add_argument("--verify-backend", choices=["z3", "dreal", "hybrid", "interval"], default="z3")
    parser.add_argument("--template", choices=["smooth", "separable-cubic", "shared-cubic", "quad-square", "hinge-relu", "monotone-relu"], default="smooth")
    parser.add_argument("--cubic-init-scale", type=float, default=0.1, help="random init scale for separable-cubic template; input remains raw (x1,x2)")
    parser.add_argument("--ce-neighborhood", type=float, default=0.0, help="CEGIS-only local cloud radius around formal counterexamples; not an input feature")
    parser.add_argument("--hidden-dim", type=int, default=3, help="number of squared hidden units; input remains raw (x1,x2)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for neural synthesis")
    parser.add_argument("--qi", type=int, default=1, help="unused (kept for CLI consistency)")
    parser.add_argument("--qj", type=int, default=1, help="unused (kept for CLI consistency)")
    args = parser.parse_args()
    set_seed(args.seed)

    print_header(
        "ex3",
        "NNT",
        "transition_persistence",
        {"solver_verify": args.verify_backend, "template": args.template, "hidden": args.hidden_dim, "max_iter": args.max_iter, "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "epsilon": args.epsilon, "cubic_init_scale": args.cubic_init_scale, "ce_neighborhood": args.ce_neighborhood},
    )
    start_time = time.time()
    B_net, success = synthesize_persistence_certificate(
        max_iterations=args.max_iter,
        train_epochs=args.epochs,
        train_lr=args.lr,
        dreal_precision=args.dreal_precision,
        epsilon=args.epsilon,
        verify_backend=args.verify_backend,
        z3_timeout_ms=args.z3_timeout_ms,
        template=args.template,
        hidden_dim=args.hidden_dim,
        grid_step=args.grid_step,
        cubic_init_scale=args.cubic_init_scale,
        ce_neighborhood=args.ce_neighborhood,
    )
    end_time = time.time()

    if success:
        print("\nCertificate summary:")
        print("-" * 70)
        summary = certificate_summary(B_net)
        print(json.dumps(summary, indent=2))
        print("-" * 70)

        sample_inits = [(21, 21), (21, 24), (24, 21), (24, 24), (22.5, 22.5)]
        for x1, x2 in sample_inits:
            B_val = B_net(torch.tensor(x1, dtype=torch.float32),
                         torch.tensor(x2, dtype=torch.float32)).item()
            print(f"  B({x1:.1f}, {x2:.1f}) = {B_val:.8f}")

        model_path = "persistence_barrier_model_2layer.pth"
        torch.save(B_net.state_dict(), model_path)
        print(f"Saved model: {model_path}")

        checkpoint_path = "persistence_barrier_checkpoint_2layer.pth"
        checkpoint = {
            'state_dict': B_net.state_dict(),
            'summary': summary,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    elapsed = end_time - start_time
    print(f"\nElapsed time: {elapsed:.4f} s")

    result = {
        "example": "ex3",
        "method": "NNT",
        "certificate_type": "transition_persistence",
        "success": bool(success),
        "seed": int(args.seed),
        "elapsed_sec": float(elapsed),
        "epsilon": float(B_net.get_epsilon()) if B_net is not None else None,
        "template": getattr(B_net, "template", None) if B_net is not None else None,
        "hidden_dim": network_size(B_net) if B_net is not None else 0,
        "solver": {"synth": "adam", "verify": args.verify_backend, "dreal_precision": args.dreal_precision if args.verify_backend == "dreal" else None},
        "formal_conditions": [
            "forall x in X0: B(x) >= 0",
            "forall x in X, f(x) in X: B(x) >= 0 => B(f(x)) >= 0",
            "forall x in X\\VF, f(x) in X: B(x) >= B(f(x))",
            "forall x in VF, f(x) in X: B(x) >= B(f(x)) + epsilon",
        ],
        "certificate": certificate_summary(B_net) if success and B_net is not None else None,
        "model_state_path": "persistence_barrier_model_2layer.pth" if success else None,
        "checkpoint_path": "persistence_barrier_checkpoint_2layer.pth" if success else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_result(bool(success), None, elapsed, str(out_path))
