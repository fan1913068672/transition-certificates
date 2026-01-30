import math
import z3
import dreal
import time

"""
case: two dimensional Kuramoto
LTL specification := G !Xu
the negation := F Xu
"""

def In_X(x1, x2):
    return x1 >= 0 and x1 <= math.pi / 9 * 8 and x2 >= 0 and x2 <= math.pi / 9 * 8


def In_X_Cond(x1_ce, x2_ce):
    return dreal.And(dreal.And(x1_ce >= 0, x1_ce <= dreal.cos(0) / 9 * 8), dreal.And(x2_ce >= 0, x2_ce <= dreal.cos(0) / 9 * 8))


def In_X0_Cond(x1_ce, x2_ce):
    return dreal.And(dreal.And(x1_ce >= 0, x1_ce <= dreal.cos(0) / 9), dreal.And(x2_ce >= 0, x2_ce <= dreal.cos(0) / 9))


def f_t(x1, x2):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega  + 1.69 + Ts * K * dreal.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega  + 1.69 + Ts * K * dreal.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p

def q_trans(q):
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise Exception


def delta(x1, x2, q):
    if q == 1:
        if In_Unsafe(x1, x2):
            return [0]
        else:
            return [1]
    else:
        return [0]


def f_m(x1, x2):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega  + 1.69 + Ts * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega  + 1.69 + Ts * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p


def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res


def In_Unsafe(x1, x2):
    return x1 >= 5 / 6 * math.pi and x1 <= 8 / 9 * math.pi or x2 >= 5 / 6 * math.pi and x2 <= 8 / 9 * math.pi


def In_Unsafe_Cond(x1_ce, x2_ce):
    return dreal.Or(dreal.And(x1_ce >= dreal.cos(0) / 6 * 5, x1_ce <= dreal.cos(0) / 9 * 8),
             dreal.And(x1_ce >= dreal.cos(0) / 6 * 5, x1_ce <= dreal.cos(0) / 9 * 8))


# def In_UnSafe_Cond(x_ce)

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
    X1_Samples = step_sample(0, 8 * math.pi / 9, 0.1)
    X2_Samples = step_sample(0, 8 * math.pi / 9, 0.1)
    Q_Samples = [0, 1]
    Y_Samples = state_space_product(X1_Samples, X2_Samples, Q_Samples)
    X01_Samples = step_sample(0, math.pi / 9, 0.1)
    X02_Samples = step_sample(0, math.pi / 9, 0.1)
    Q0_Samples = [1]
    Y0_Samples = state_space_product(X01_Samples, X02_Samples, Q0_Samples)
    Qacc_Samples = [0]
    Yu_Samples = state_space_product(X1_Samples, X2_Samples, Qacc_Samples)

    def Bp_t(x1, x2, q):
        res = c[0] + c[1] * x1 + c[2] * x2 ** 2 + c[3] * x1 * x2 + c[4] * x1 ** 3 + c[5] * q + c[6] * q ** 2 + c[7] * q * x1 ** 2 + c[8] * x2 * q
        return z3.simplify(res)

    for x01, x02, q0 in Y0_Samples:
        s.add(Bp_t(x01, x02, q0) >= 0)

    for x1, x2, qacc in Yu_Samples:
        s.add(Bp_t(x1, x2, qacc) < 0)

    for x1, x2, q in Y_Samples:
        x1p, x2p = f_m(x1, x2)
        qp_list = delta(x1, x2, q)
        for qp in qp_list:
            s.add(z3.Implies(Bp_t(x1, x2, q) >= 0, Bp_t(x1p, x2p, qp) >= 0))

    MAX_ITER = 1000
    iter = 0

    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.0001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x1_ce = dreal.Variable('x1_ce')
    x2_ce = dreal.Variable('x2_ce')
    ce_solver.DeclareVariable(x1_ce, 0, dreal.cos(0) / 9 * 8)
    ce_solver.DeclareVariable(x2_ce, 0, dreal.cos(0) / 9 * 8)
    while s.check() == z3.sat and iter < MAX_ITER:
        ce_flag = False
        m = s.model()
        c_m = [t2float(m[i]) for i in c]
        # print(f"#{iter}:")
        # for i, cc in enumerate([m[item] for item in c]):
        #     print(f"c{i}={t2float(cc)}")

        def Bp_c(x1, x2, q):
            return c_m[0] + c_m[1] * x1 + c_m[2] * x2 ** 2 + c_m[3] * x1 * x2 + c_m[4] * x1 ** 3 + c_m[5] * q + c_m[6] * q ** 2 + c_m[7] * q * x1 ** 2 + c_m[8] * x2 * q

        ce_solver.Push(2)
        ce_solver.Assert(In_X0_Cond(x1_ce, x2_ce))
        ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) < 0)
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print("A counterexample to the non-negativity property in the initial state")
            x1_m = ce_model[x1_ce].mid()
            x2_m = ce_model[x2_ce].mid()
            print(x1_m, x2_m, 1, Bp_c(x1_m, x2_m, 1))
            ce_flag = True
            s.add(Bp_t(x1_m, x2_m, 1) >= 0)
        else:
            print("The non-negativity property in the initial state has passed the test.")
        ce_solver.Pop(2)
        ce_solver.Push(2)
        ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
        ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) >= 0)
        ce_model = ce_solver.CheckSat()
        if ce_model != None:
            print("A counterexample to the negativity property in the unsafe states")
            x1_m = ce_model[x1_ce].mid()
            x2_m = ce_model[x2_ce].mid()
            print(x1_m, x2_m, 0, Bp_c(x1_m, x2_m, 0))
            ce_flag = True
            s.add(Bp_t(x1_m, x2_m, 0) < 0)
        else:
            print("The negativity property in the unsafe state has passed the test.")
        ce_solver.Pop(2)

        tnn_flag = False
        ce_solver.Push(3)
        ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
        ce_solver.Assert(In_X_Cond(*f_t(x1_ce, x2_ce)))
        ce_solver.Assert(Bp_c(x1_ce, x2_ce, 1) >= 0)
        qp_list = q_trans(1)
        for qp in qp_list:
            ce_solver.Push(2)
            if qp == 0:
                ce_solver.Assert(In_Unsafe_Cond(x1_ce, x2_ce))
                ce_solver.Assert(Bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            elif qp == 1:
                ce_solver.Assert(dreal.Not(In_Unsafe_Cond(x1_ce, x2_ce)))
                ce_solver.Assert(Bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            else:
                raise Exception
            ce_model = ce_solver.CheckSat()
            if ce_model != None:
                print("A counterexample to the non-negativity property through a transition.")
                ce_flag = True
                tnn_flag = True
                x1_m = ce_model[x1_ce].mid()
                x2_m = ce_model[x2_ce].mid()
                x1_mn, x2_mn = f_m(x1_m, x2_m)
                print(f"{x1_m}, {x2_m}, {0}, {Bp_c(x1_m, x2_m, 0)}")
                print(f"{x1_mn}, {x2_mn}, {qp}, {Bp_c(x1_mn, x2_mn, qp)}")
                s.add(z3.Implies(Bp_t(x1_m, x2_m, 0) >= 0, Bp_t(x1_mn, x2_mn, qp) >= 0))
            ce_solver.Pop(2)
        ce_solver.Pop(3)

        ce_solver.Push(3)
        ce_solver.Assert(In_X_Cond(x1_ce, x2_ce))
        ce_solver.Assert(In_X_Cond(*f_t(x1_ce, x2_ce)))
        ce_solver.Assert(Bp_c(x1_ce, x2_ce, 0) >= 0)
        qp_list = q_trans(0)
        for qp in qp_list:
            ce_solver.Push(1)
            if qp == 0:
                ce_solver.Assert(Bp_c(*f_t(x1_ce, x2_ce), qp) < 0)
            else:
                raise Exception
            ce_model = ce_solver.CheckSat()
            if ce_model != None:
                print("A counterexample to the non-negativity property through a transition.")
                ce_flag = True
                tnn_flag = True
                x1_m = ce_model[x1_ce].mid()
                x2_m = ce_model[x2_ce].mid()
                x1_mn, x2_mn = f_m(x1_m, x2_m)
                print(f"{x1_m}, {x2_m}, {1}, {Bp_c(x1_m, x2_m, 1)}")
                print(f"{x1_mn}, {x2_mn}, {qp}, {Bp_c(x1_mn, x2_mn, qp)}")
                s.add(z3.Implies(Bp_t(x1_m, x2_m, 1) >= 0, Bp_t(x1_mn, x2_mn, qp) >= 0))
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
B(x1, x2, q) = c_m[0] + c_m[1] * x1 + c_m[2] * x2 ** 2 + c_m[3] * x1 * x2 + c_m[4] * x1 ** 3 + c_m[5] * q + c_m[6] * q ** 2 + c_m[7] * q * x1 ** 2 + c_m[8] * x2 * q
Parameters:
c0 = -1
c1 = 0
c2 = 0
c3 = 1/8
c4 = 0
c5 = 0
c6 = 3/2
c7 = -3/32
c8 = -153/512
Time taken to synthesize the safety certificate: 7.2672 seconds
"""