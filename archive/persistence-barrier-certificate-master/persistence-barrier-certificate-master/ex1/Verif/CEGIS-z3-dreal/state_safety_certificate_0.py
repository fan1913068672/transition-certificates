import math
import z3
import dreal
import time

"""
case: one dimensional Kuramoto
LTL specification := G !Xu
the negation := F Xu
"""
PI = 3.1415926
def In_X_Cond(x_ce):
    return dreal.And(x_ce >= 0, x_ce <= PI * 2)


def In_X0_Cond(x_ce):
    return dreal.And(x_ce >= 4 * PI / 9, x_ce <= 5 * PI / 9)


def f_t(x_ce):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x_ce + Ts * Omega + Ts * K * dreal.sin(-x_ce) - 0.532 * x_ce ** 2 + 1.69


def q_trans(q):
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise Exception


def delta(x, q):
    if q == 1:
        if In_Unsafe(x):
            return [0]
        else:
            return [1]
    else:
        return [0]


def f_m(x):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    xp = x + Ts * Omega + Ts * K * math.sin(-x) - 0.532 * x ** 2 + 1.69
    return xp


def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res


def In_Unsafe(x):
    return x >= 7 / 9 * PI and x <= 8 / 9 * PI


def In_Unsafe_Cond(x_ce):
    return dreal.And(x_ce >= PI / 9 * 7, x_ce <= PI / 9 * 8)


def t2float(a, precision=14):
    s = a.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)


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


def reachability_10():
    print("Synthesizing a state safety certificate for 0")
    cc_flag = False
    c = [z3.Real(f'c{i}') for i in range(0, 9)]
    s = z3.SolverFor("QF_NRA")
    X_Samples = step_sample(0, 2 * PI, 0.01)
    Q_Samples = [0, 1]
    Y_Samples = state_space_product(X_Samples, Q_Samples)
    X0_Samples = step_sample(PI * 4 / 9, PI * 5 / 9, 0.01)
    Q0_Samples = [1]
    Y0_Samples = state_space_product(X0_Samples, Q0_Samples)
    Qacc_Samples = [0]
    Yu_Samples = state_space_product(X_Samples, Qacc_Samples)

    def Bp_t(x, q):
        res = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4 + c[5] * q + c[6] * q ** 2 + c[
            7] * q ** 3 + c[8] * x * q
        return z3.simplify(res)

    for x0, q0 in Y0_Samples:
        s.add(Bp_t(x0, q0) >= 0)

    for x, qacc in Yu_Samples:
        s.add(Bp_t(x, qacc) < 0)

    for x, q in Y_Samples:
        xp = f_m(x)
        qp_list = delta(x, q)
        for qp in qp_list:
            s.add(z3.Implies(Bp_t(x, q) >= 0, Bp_t(xp, qp) >= 0))

    MAX_ITER = 1000
    iter = 0

    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.0001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x_ce = dreal.Variable('x_ce')
    ce_solver.DeclareVariable(x_ce, 0, 2 * PI)
    while s.check() == z3.sat and iter < MAX_ITER:
        ce_flag = False
        m = s.model()
        c_m = [t2float(m[i]) for i in c]
        print(f"#{iter}:")
        for i, cc in enumerate([m[item] for item in c]):
            print(f"c{i}={t2float(cc)}")

        def Bp_c(x, q):
            return c_m[0] + c_m[1] * x + c_m[2] * x ** 2 + c_m[3] * x ** 3 + c_m[4] * x ** 4 + c_m[5] * q + c_m[
                6] * q ** 2 + c_m[7] * q ** 3 + c_m[8] * x * q

        ce_solver.Push(2)
        ce_solver.Assert(In_X0_Cond(x_ce))
        ce_solver.Assert(Bp_c(x_ce, 1) < 0)
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print("A counterexample to the non-negativity property in the initial state")
            print(ce_model[x_ce].mid(), Bp_c(ce_model[x_ce].mid(), 1))
            ce_flag = True
            s.add(Bp_t(ce_model[x_ce].mid(), 1) >= 0)
        else:
            print("The non-negativity property in the initial state has passed the test.")
        ce_solver.Pop(2)

        ce_solver.Push(2)
        ce_solver.Assert(In_X_Cond(x_ce))
        ce_solver.Assert(Bp_c(x_ce, 0) >= 0)
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print("A counterexample to the negativity property in the unsafe states")
            print(ce_model[x_ce].mid(), Bp_c(ce_model[x_ce].mid(), 0))
            ce_flag = True
            s.add(Bp_t(ce_model[x_ce].mid(), 0) < 0)
        else:
            print("The negativity property in the unsafe state has passed the test.")
        ce_solver.Pop(2)

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
                print("A counterexample to the non-negativity property through a transition.")
                ce_flag = True
                tnn_flag = True
                x_ce_m = ce_model[x_ce].mid()
                x_ce_p_m = f_m(x_ce_m)
                print(f"{Bp_c(x_ce_m, 0)}")
                print(f"{Bp_c(x_ce_p_m, qp)}")
                s.add(z3.Implies(Bp_t(x_ce_m, 0) >= 0, Bp_t(x_ce_p_m, qp) >= 0))
            ce_solver.Pop(2)
        ce_solver.Pop(3)

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
                print("A counterexample to the non-negativity property through a transition.")
                ce_flag = True
                tnn_flag = True
                x_ce_m = ce_model[x_ce].mid()
                x_ce_p_m = f_m(x_ce_m)
                print(f"{Bp_c(x_ce_m, 1)}")
                print(f"{Bp_c(x_ce_p_m, qp)}")
                s.add(z3.Implies(Bp_t(x_ce_m, 1) >= 0, Bp_t(x_ce_p_m, qp) >= 0))
            ce_solver.Pop(1)
        ce_solver.Pop(3)
        if not tnn_flag:
            print("The non-negativity property through a transition has passed the test.")
        if ce_flag is False:
            print("Resulting Certificate：")
            cc_flag = True
            for i, cc in enumerate([m[item] for item in c]):
                print(f"c{i}={cc}")
            break
        iter = iter + 1

    if iter > MAX_ITER:
        print("Exceeded maximum number of iterations")
    else:
        if not cc_flag:
            print("Unable to synthesize candidate state safety certificate")


start_time = time.time()
reachability_10()
end_time = time.time()
print(f"Time taken to synthesize state safety certificate: {end_time - start_time:.4f} seconds")

"""
The goal is to check whether all traces fail to transition from q1 to q0.
Experiment results on 2025/03/15:
Certificate template:
B(x, q) = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4 + c[5] * q +
c[6] * q ** 2 + c[7] * q ** 3 + c[8] * x * q
Certificate results:
c0 = -1
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 2
c7 = 0
c8 = -7/16
Time taken to synthesize the safety certificate: 4.2516 seconds
"""