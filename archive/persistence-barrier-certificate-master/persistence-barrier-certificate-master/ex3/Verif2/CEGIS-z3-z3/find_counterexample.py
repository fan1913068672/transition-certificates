import time
import z3

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

def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1/s)), int(b*int(1/s))+1):
        res.append(i * s)
    return res

def f_m(x1, x2):
    alpha = 0.004
    theta = 0.01
    Te = 0
    Th = 40
    mu = 0.15

    def u(x_curr):
        return 0.59 - 0.011 * x_curr

    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te
    return x1_next, x2_next

def f_t(x1p, x2p):
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

def In_X0(x1, x2):
    return x1 >= 21 and x1 <= 24 and x2 >=21 and x2 <=24

def In_X0_Cond(x1, x2):
    return z3.And(x1 >= 21, x1 <= 24, x2 >=21, x2 <=24)

def In_X_Cond(x1p, x2p):
    return z3.And(x1p >= 20, x1p <= 34, x2p >= 20, x2p <= 34)


def delta_m(x1, x2, q):
    if q == 0:
        return [1]
    if q == 1:
        if In_X0(x1, x2):
            return [2]
        else:
            return [3]
    if q == 2:
        return [0]
    if q == 3:
        return [3]

def delta_t(x1, x2, q):
    if q == 0:
        return [[],], [1]
    if q == 1:
        return [[In_X0_Cond(x1, x2)], [z3.Not(In_X0_Cond(x1, x2))]], [2, 3]
    if q == 2:
        return [[],], [0]
    if q == 3:
        return [[], ], [3]

def Q_Trans_Acc_Cond(x1, x2, q1, q2):
    if q1 == 1 and q2 == 2:
        return In_X_Cond(x1, x2)
    else:
        raise Exception

def Q_Trans_Acc(x1, x2, q1, q2):
    if In_X0(x1, x2) and q1 == 1 and q2 == 2:
        return True
    else:
        return False

def state_safety(qu):
    print(f"Synthesizing a state safety certificate for {qu}")
    certificate_flag = False
    s = z3.SolverFor("QF_NRA")

    X1_samples = step_sample(20, 34, 1)
    X2_samples = step_sample(20, 34, 1)
    Y_Samples = state_space_product(X1_samples, X2_samples, [0, 1, 2, 3])

    Yu_Samples = state_space_product(X1_samples, X2_samples, [2])
    c = [z3.Real(f'c{i}') for i in range(0, 9)]
    def Bp_t(x1, x2, q):
        res = c[0] + c[1] * x1 + c[2] * x2 ** 2 + c[3] * x1 * x2 + c[4] * x1 ** 3 + c[5] * q + c[6] * q ** 2 + c[7] * q * x1 ** 2 + c[8] * x2 * q
        return z3.simplify(res)

    q0 = 0
    X01_samples = step_sample(21, 24, 0.1)
    X02_samples = step_sample(21, 24, 0.1)
    Y0_Samples = state_space_product(X01_samples, X02_samples, [q0])
    for x10, x20, q0 in Y0_Samples:
        s.add(Bp_t(x10, x20, q0) >= 0)

    for x1, x2, qacc in Yu_Samples:
        s.add(Bp_t(x1, x2, qacc) < 0)

    for x1, x2, q in Y_Samples:
        x1p, x2p = f_m(x1, x2)
        qp_list = delta_m(x1, x2, q)
        for qp in qp_list:
            s.add(z3.Implies(Bp_t(x1, x2, q) >= 0, Bp_t(x1p, x2p, qp) >= 0))

    MAX_ITER = 1000
    iter = 0
    ce_solver = z3.SolverFor("QF_NRA")
    x1_ce = z3.Real('x1_ce')
    x2_ce = z3.Real('x2_ce')
    while s.check() == z3.sat and iter < MAX_ITER:
        ce_flag = False
        m = s.model()
        c_m = [m[i] for i in c]
        def Bp_c(x1, x2, q):
            nonlocal c_m
            return c_m[0] + c_m[1] * x1 + c_m[2] * x2 ** 2 + c_m[3] * x1 * x2 + c_m[4] * x1 ** 3 + c_m[5] * q + c_m[6] * q ** 2 + c_m[7] * q * x1 ** 2 + c_m[8] * x2 * q

        ce_solver.push()
        ce_solver.add(In_X0_Cond(x1_ce, x2_ce))
        ce_solver.add(Bp_c(x1_ce, x2_ce, q0) < 0)

        if ce_solver.check() == z3.sat:
            print("A counterexample to the non-negativity property in the initial state")
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            ce_flag = True
            s.add(Bp_t(x1_m, x2_m, q0) >= 0)
        else:
            print("The non-negativity property in the initial state has passed the test.")
        ce_solver.pop()

        ce_solver.push()
        ce_solver.add(In_X_Cond(x1_ce, x2_ce))
        ce_solver.add(Bp_c(x1_ce, x2_ce, qu) >= 0)
        if ce_solver.check() == z3.sat:
            print("A counterexample to the negativity property in the unsafe states")
            ce_flag = True
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            s.add(Bp_t(x1_m, x2_m, qu) < 0)
        else:
            print("The negativity property in the unsafe state has passed the test.")
        ce_solver.pop()

        tnn_flag = True
        ce_solver.push()
        ce_solver.add(In_X_Cond(x1_ce, x2_ce))
        for q in [0, 1, 2, 3]:
            ce_solver.add(Bp_c(x1_ce, x2_ce, q) >= 0)
            conds_list, qp_list = delta_t(x1_ce, x2_ce, q)
            for conds, qp in zip(conds_list, qp_list):
                ce_solver.push()
                for cond in conds:
                    ce_solver.add(cond)
                ce_solver.add(Bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
                if ce_solver.check() == z3.sat:
                    print("A counterexample to the non-negativity property through a transition.")
                    ce_flag = True
                    tnn_flag = False
                    ce_model = ce_solver.model()
                    x1_m = ce_model[x1_ce]
                    x2_m = ce_model[x2_ce]
                    x1_mn, x2_mn = f_m(x1_m, x2_m)
                    s.add(z3.Implies(Bp_t(x1_m, x2_m, q) >= 0, Bp_t(x1_mn, x2_mn, qp) >= 0))
                ce_solver.pop()
        ce_solver.pop()
        if tnn_flag:
            print("the non-negativity property through a transition has passed the test.")

        if ce_flag is False:
            print(f"State safety certificate for {qu} has found：")
            for i, cc in enumerate([m[item] for item in c]):
                print(f"c{i}={cc}")
            return True
        iter = iter + 1

    if iter > MAX_ITER:
        print("Exceeded maximum number of iterations")
    else:
        if not certificate_flag:
            print("Unable to synthesize candidate state safety certificate")
    return False

def transition_persistence(q1, q2):
    print(f"Synthesize a transition persistent certificate for {q1} to {q2}")
    certificate_flag = False
    c = [z3.Real(f'c{i}') for i in range(7)]
    epsilon = z3.Real('epsilon')
    s = z3.SolverFor("QF_NRA")
    X1_samples = step_sample(20, 34, 1)
    X2_samples = step_sample(20, 34, 1)
    Samples = state_space_product(X1_samples, X2_samples)

    def Bp_t(x1, x2):
        res = c[0] + c[1] * z3.If(In_X0_Cond(x1, x2), 1, 0) +  c[2] * x1 * z3.If(In_X0_Cond(x1, x2), 1, 0) + c[3] * x2 * z3.If(In_X0_Cond(x1, x2), 1, 0) + c[4] * z3.If(x1 > x2, x1, x2) + c[5] * x1 ** 2 + c[6] * x2 ** 2
        return z3.simplify(res)


    X01_samples = step_sample(21, 24, 0.1)
    X02_samples = step_sample(21, 24, 0.1)
    X0_Samples = state_space_product(X01_samples, X02_samples)
    for x10, x20 in X0_Samples:
        s.add(Bp_t(x10, x20) >= 0)

    for x1, x2 in Samples:
        x1n, x2n = f_m(x1, x2)
        s.add(z3.Implies(Bp_t(x1, x2) >= 0, Bp_t(x1n, x2n) >= 0))
        if Q_Trans_Acc(x1, x2, q1, q2):
            s.add(Bp_t(x1, x2) >= Bp_t(x1n, x2n) + epsilon)
        else:
            s.add(Bp_t(x1, x2) >= Bp_t(x1n, x2n))

    s.add(epsilon > 0)

    ce_solver = z3.SolverFor("QF_NRA")
    x1_ce = z3.Real('x1_ce')
    x2_ce = z3.Real('x2_ce')
    iter = 0
    MAX_ITER = 1000
    while s.check() == z3.sat and iter <= MAX_ITER:
        ce_flag = False
        candidate_model = s.model()
        e_m = candidate_model[epsilon]
        c_m = [candidate_model[item] for item in c]
        def Bp_c(x1, x2):
            nonlocal c_m
            res = c_m[0] + c_m[1] * z3.If(In_X0_Cond(x1, x2), 1, 0) +  c_m[2] * x1 * z3.If(In_X0_Cond(x1, x2), 1, 0) + c_m[3] * x2 * z3.If(In_X0_Cond(x1, x2), 1, 0) + c_m[4] * z3.If(x1 > x2, x1, x2) + c_m[5] * x1 ** 2 + c_m[6] * x2 ** 2
            return z3.simplify(res)

        ce_solver.push()
        ce_solver.add(In_X0_Cond(x1_ce, x2_ce))
        ce_solver.add(Bp_c(x1_ce, x2_ce) < 0)

        if ce_solver.check() == z3.sat:
            print("A counterexample to the non-negativity property in the initial state")
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            ce_flag = True
            s.add(Bp_t(x1_m, x2_m) >= 0)
        else:
            print("The non-negativity property in the initial state has passed the test.")
        ce_solver.pop()

        ce_solver.push()
        ce_solver.add(In_X_Cond(x1_ce, x2_ce))
        ce_solver.add(Bp_c(x1_ce, x2_ce) >= 0)
        ce_solver.add(Bp_c(*f_t(x1_ce, x2_ce)) < 0)
        if ce_solver.check() == z3.sat:
            print("The non-negativity preservation property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            s.add(z3.Implies(Bp_t(x1_m, x2_m) >= 0, Bp_t(*f_t(x1_m, x2_m)) >= 0))
        else:
            print(
                "The non-negativity preservation property check for the certificate value across transitions has passed.")

        ce_solver.push()
        ce_solver.add(In_X_Cond(x1_ce, x2_ce))
        ce_solver.add(z3.Not(Q_Trans_Acc_Cond(x1_ce, x2_ce, q1, q2)))
        ce_solver.add(Bp_c(x1_ce, x2_ce) < Bp_c(*f_t(x1_ce, x2_ce)))
        if ce_solver.check() == z3.sat:
            print("The non-increasing property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            s.add(Bp_t(x1_m, x2_m) >= Bp_t(*f_t(x1_m, x2_m)))
        else:
            print("The non-increasing property check for the certificate value across transitions has passed.")
        ce_solver.pop()

        ce_solver.push()
        ce_solver.add(In_X_Cond(x1_ce, x2_ce))
        ce_solver.add(Q_Trans_Acc_Cond(x1_ce, x2_ce, q1, q2))
        ce_solver.add(Bp_c(x1_ce, x2_ce) < Bp_c(*f_t(x1_ce, x2_ce)) + e_m)
        if ce_solver.check() == z3.sat:
            print("The decreasing property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            x1_m = ce_model[x1_ce]
            x2_m = ce_model[x2_ce]
            s.add(Bp_t(x1_m, x2_m) >= Bp_t(*f_t(x1_m, x2_m)) + epsilon)
        else:
            print("The decreasing property check for the certificate value across transitions has passed.")
        ce_solver.pop()
        if not ce_flag:
            print("Certificate found!")
            for idx, item in enumerate(c_m):
                print(f"c[{idx}] = {item}")
            print(f"epsilon={e_m}")
            return True
        iter = iter + 1

    if iter > MAX_ITER:
        print("Exceeded the maximum number of iterations.")
    else:
        if not certificate_flag:
            print("Unable to synthesize candidate certificate.")
    return False

def main():
    flag1 = state_safety(2)
    flag2 = transition_persistence(1, 2)
    return flag1 or flag2

start = time.time()
flag1 = state_safety(2)
end = time.time()
print("flag: ", flag1)
print("Time cost: ", end-start, "s")
# start = end
# flag2 = transition_persistence(1, 2)
# end = time.time()
# print("flag: ", flag2)
# print("Time cost: ", end-start, "s")

"""
2025/03/30
Synthesize a transition persistent certificate for 1 to 2
The non-negativity property in the initial state has passed the test.
The non-negativity preservation property check for the certificate value across transitions has passed.
The non-increasing property check for the certificate value across transitions has passed.
The decreasing property check for the certificate value across transitions has passed.
Certificate found!
c[0] = 0
c[1] = 25
c[2] = 0
c[3] = -1
c[4] = 0
c[5] = 0
c[6] = 0
epsilon=1/2
flag:  True
Time cost:  6.625749349594116 s
"""