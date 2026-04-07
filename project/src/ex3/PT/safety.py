import time

from z3 import *

def InAcc(q):
    return q == 1

def InQ0(q):
    return q in {0}

def In_X_Cond(x1p, x2p):
    return And(x1p >= 20, x1p <= 34, x2p >= 20, x2p <= 34)

def In_X(x1, x2):
    return x1 >= 20 and x1 <= 34 and x2 >= 20 and x2 <= 34

def Q_Trans_Cond(x1p, x2p, qi, qj):
    if qi == 0 and qj == 1:
        return In_X0_Cond(x1p, x2p)
    if qi == 0 and qj == 2:
        return Not(In_X0_Cond(x1p, x2p))
    if qi == 1 and qj == 1 or qi == 2 and qj ==2:
        return True
    return False

def In_X0_Cond(x1p, x2p):
    return And(x1p >= 21, x1p <= 24, x2p >= 21, x2p <= 24)

def In_X0(x1, x2):
    return x1 >= 21 and x1 <= 24 and x2 >=21 and x2 <=24

def f_cond(x1p, x2p):
    alpha = 0.004
    theta = 0.01
    Te = 0
    Th = 40
    mu = 0.15
    def u(x_curr):
        return 0.59 - 0.011 * x_curr
    x1_next = (1 - 2 * alpha - theta - mu * u(x1p)) * x1p + x2p * alpha + mu * Th * u(x1p) + theta * Te
    x2_next = x1p * alpha + (1 - 2 * alpha - theta - mu * u(x2p)) * x2p + mu * Th * u(x2p) + theta * Te
    return x1_next, x2_next

def f(x1, x2, q):
    alpha = 0.004
    theta = 0.01
    Te = 0
    Th = 40
    mu = 0.15

    def u(x_curr):
        return 0.59 - 0.011 * x_curr

    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te

    if q == 0:
        if In_X0(x1, x2):
            return x1_next, x2_next, [1]
        else:
            return x1_next, x2_next, [2]
    elif q == 1:
        return x1_next, x2_next, [1]
    elif q == 2:
        return x1_next, x2_next, [2]

def L(x1_val, x2_val):
    cond1 = x1_val >= 21 and x1_val <= 24 and x2_val >= 21 and x2_val <= 24
    cond2 = x1_val >= 20 and x1_val <= 26 and x2_val >= 20 and x2_val <= 26
    if cond1 and cond2:
        return 3 #{a,b}
    if cond1 and not cond2:
        return 2 #{a}
    if not cond1 and cond2:
        return 1 #{b}
    if not cond1 and not cond2:
        return 0 #{}

def B_Cond(x1p, x2p):
    return And(x1p >= 20, x1p <= 26, x2p >= 20, x2p <= 26)

def In_VF(x1_val, x2_val):
    return x1_val >= 20 and x1_val <= 26 and x2_val >= 20 and x2_val <= 26

def space_product(s1, s2):
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

def check_trans(x1, x2, qi, qj):
    if L(x1, x2) >= 2 and qi == 0 and qj == 1:
        return True
    if L(x1, x2) < 2 and qi == 0 and qj == 2:
        return True
    if qi == 1 and qj == 1:
        return True
    if qi == 2 and qj == 2:
        return True
    return False

def step_sample(a, b, s):
    res = []
    for i in range(a * int(1/s), b*int(1/s)+1):
        res.append(i * s)
    return res
def Not_Acc_Cond(x1p, x2p, qi, qj):
    if qi != 1 or qj != 1:
        return True
    return Not(B_Cond(x1p, x2p))

def t2float(a, precision=14):
    s = a.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)

def unreachability_synthesis(qi):
    print(f"Synthesize a state safety certificate for {qi}")
    c = [Real(f'c{i}') for i in range(0, 12)]

    s = SolverFor("QF_NRA")
    X1_samples = step_sample(20, 34, 1)
    X2_samples = step_sample(20, 34, 1)
    Y_Samples = state_space_product(X1_samples, X2_samples, [0, 1, 2])

    Yu_Samples = state_space_product(X1_samples, X2_samples, [1])

    X01_samples = step_sample(21, 24, 0.1)
    X02_samples = step_sample(21, 24, 0.1)
    Q0_samples = [0]
    Y0_Samples = state_space_product(X01_samples, X02_samples, Q0_samples)

    def Bp(x1, x2, q):
        res = c[0] + c[1] * If(In_X0_Cond(x1, x2), 1, 0) + c[2] * If(B_Cond(x1, x2), 1, 0) + c[3] * x1 * If(In_X0_Cond(x1, x2), 1, 0) + c[4] * x2 * If(In_X0_Cond(x1, x2), 1, 0) + c[5] * x1 * If(B_Cond(x1, x2), 1, 0) + c[6] * x2 * If(B_Cond(x1, x2), 1, 0) + c[7] * If(x1 > x2, x1, x2) + c[8] * x1 ** 2 + c[9] * x2 ** 2 + c[10] * q + c[11] * q**2
        return simplify(res)

    for x1, x2, q0 in Y0_Samples:
        s.add(Bp(x1, x2, q0) >= 0)

    for x1, x2, qacc in Yu_Samples:
        s.add(Bp(x1, x2, qacc) < 0)

    for x1, x2, qacc in Y_Samples:
        x1n, x2n, qn_list = f(x1, x2, qacc)
        for qp in qn_list:
            s.add(Implies(Bp(x1, x2, qacc) >= 0, Bp(x1n, x2n, qp) >= 0) )

    if s.check() == sat:
        candidate_model = s.model()
        print(f"Candidate solution #{iter + 1}:")
        for idx, item in enumerate(c):
            print(f"c{idx}={t2float(candidate_model[item])}")
    else:
        print("Unable to synthesize candidate certificate")
start_time = time.time()
unreachability_synthesis(1)
end_time = time.time()
print(f"Time cost:{end_time - start_time}")

"""
March 15, 2025
Unable to synthesize candidate certificate
Time cost: 16.244112968444824 (seconds)
"""
