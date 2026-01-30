from z3 import *
import time

def InAcc(q):
    """Check if the automaton state q is an accepting state"""
    return q == 1

def InQ0(q):
    """Check if the automaton state q belongs to the initial states"""
    return q in {0}

def In_X_Cond(x1p, x2p):
    return And(x1p >= 20, x1p <= 34, x2p >= 20, x2p <= 34)

def In_X(x1, x2):
    return x1 >= 20 and x1 <= 34 and x2 >= 20 and x2 <= 34

def Q_Trans_Cond(x1p, x2p, qi, qj):
    """
    Ensure that a transition from qi to qj is possible according to (x1p, x2p)
    """
    if qi == 0 and qj == 1:
        return In_X0_Cond(x1p, x2p)
    if qi == 0 and qj == 2:
        return Not(In_X0_Cond(x1p, x2p))
    if qi == 1 and qj == 1 or qi == 2 and qj ==2:
        return True
    return False

def In_X0_Cond(x1p, x2p):
    """(x1p, x2p) is the initial state"""
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


def f(x1, x2):
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



def L(x1_val, x2_val):
    """ labelling function """
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

def cegeis_synthesis(qi, qj):
    print(f"Synthesize a transition persistent certificate for {qi} to {qj}")
    cc_flag = False
    c = [Real(f'c{i}') for i in range(0, 10)]
    epsilon = Real('epsilon')
    s = SolverFor("QF_NRA")
    X1_samples = step_sample(20, 34, 1)
    X2_samples = step_sample(20, 34, 1)
    Samples = state_space_product(X1_samples, X2_samples)
    X01_samples = step_sample(21, 24, 0.1)
    X02_samples = step_sample(21, 24, 0.1)
    X0_Samples = state_space_product(X01_samples, X02_samples)
    def Bp(x1, x2):
        res = c[0] + c[1] * If(In_X0_Cond(x1, x2), 1, 0) + c[2] * If(B_Cond(x1, x2), 1, 0) + c[3] * x1 * If(In_X0_Cond(x1, x2), 1, 0) + c[4] * x2 * If(In_X0_Cond(x1, x2), 1, 0) + c[5] * x1 * If(B_Cond(x1, x2), 1, 0) + c[6] * x2 * If(B_Cond(x1, x2), 1, 0) + c[7] * If(x1 > x2, x1, x2) + c[8] * x1 ** 2 + c[9] * x2 ** 2
        return simplify(res)
    for x1, x2 in X0_Samples:
        if check_trans(x1, x2, qi, qj):
            s.add(Bp(x1, x2) >= 0)
    for x1, x2 in Samples:
        if check_trans(x1, x2, qi, qj):
            x1n, x2n = f(x1, x2)
            if In_X(x1n, x2n):
                s.add(Implies(Bp(x1, x2) >= 0, Bp(x1n, x2n) >= 0))
                if qi == 1 and qj == 1 and In_VF(x1,x2):
                    s.add(Bp(x1, x2) >= Bp(x1n, x2n) + epsilon)
                else:
                    s.add(Bp(x1, x2) >= Bp(x1n, x2n))
    s.add(epsilon > 0)

    ce_solver = SolverFor("QF_NRA")
    x1_ce = Real('x1_ce')
    x2_ce = Real('x2_ce')
    x1_ce_next, x2_ce_next = f_cond(x1_ce, x2_ce)
    iter = 0
    MAX_ITER = 1000

    while s.check() == sat and iter <= MAX_ITER:
        ce_flag = False
        candidate_model = s.model()
        print(f"#{iter + 1} candidate solution:")
        print(f"ε={t2float(candidate_model[epsilon])}")
        for idx, item in enumerate(c):
            print(f"c{idx}={t2float(candidate_model[item])}")

        def Bp_candidate(x1p, x2p):
            c_candidate = [candidate_model[item] for item in c]
            res = c_candidate[0] + c_candidate[1] * If(In_X0_Cond(x1p, x2p), 1, 0) + c_candidate[2] * If(B_Cond(x1p, x2p), 1, 0) + c_candidate[3] * x1p * If(In_X0_Cond(x1p, x2p), 1, 0) + c_candidate[4] * x2p * If(In_X0_Cond(x1p, x2p), 1, 0) + c_candidate[5] * x1p * If(B_Cond(x1p, x2p), 1, 0) + c_candidate[6] * x2p * If(B_Cond(x1p, x2p), 1, 0) + c_candidate[7] * If(x1p > x2p, x1p, x2p) + c_candidate[8] * x1p ** 2 + c_candidate[9] * x2p ** 2
            return simplify(res)

        def Bp_candidate2(x1, x2):
            c_candidate = [t2float(candidate_model[item]) for item in c]
            res = c_candidate[0] + c_candidate[1] * In_X0(x1, x2) + c_candidate[2] * In_VF(x1, x2) + c_candidate[3] * x1 * In_X0(x1, x2) + c_candidate[4] * x2 * In_X0(x1, x2) + c_candidate[5] * x1 * In_VF(x1, x2) + c_candidate[6] * x2 * In_VF(x1, x2) + c_candidate[7] * max(x1, x2) + c_candidate[8] * x1 ** 2 + c_candidate[9] * x2 ** 2
            return res

        ce_solver.push()
        ce_solver.add(
            Q_Trans_Cond(x1_ce, x2_ce, qi, qj),
            In_X0_Cond(x1_ce, x2_ce),
            Bp_candidate(x1_ce, x2_ce) < 0)
        if ce_solver.check() == sat:
            print("The non-negativity property of the certificate value at the initial state is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            tmp_x1_ce_model, tmp_x2_ce_model = t2float(ce_model[x1_ce]), t2float(ce_model[x2_ce])
            tmp_bp = Bp_candidate2(tmp_x1_ce_model, tmp_x2_ce_model)
            print(f"x1_ce={tmp_x1_ce_model}, x2_ce={tmp_x2_ce_model}, Bp_c={tmp_bp}")
            s.add(Bp(tmp_x1_ce_model, tmp_x2_ce_model) >= 0)
        else:
            print("The non-negativity property check for the certificate value at the initial state has passed.")
        ce_solver.pop()

        ce_solver.push()
        ce_solver.add(
            Q_Trans_Cond(x1_ce, x2_ce, qi, qj),
            In_X_Cond(x1_ce, x2_ce),
            In_X_Cond(x1_ce_next, x2_ce_next),
            Not_Acc_Cond(x1_ce, x2_ce, qi, qj),
            Bp_candidate(x1_ce, x2_ce) >= 0,
            Bp_candidate(x1_ce_next, x2_ce_next) < 0)
        if ce_solver.check() == sat:
            print("The non-negativity preservation property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            tmp_x1_ce_model, tmp_x2_ce_model = t2float(ce_model[x1_ce]), t2float(ce_model[x2_ce])
            tmp_x1_ce_model_next, tmp_x2_ce_model_next = f(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp = Bp_candidate2(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp2 = Bp_candidate2(tmp_x1_ce_model_next, tmp_x2_ce_model_next)
            print(f"x1_ce={tmp_x1_ce_model}, x2_ce={tmp_x2_ce_model}, Bp_c={tmp_bp}, Bp_c'={tmp_bp2}")
            s.add(Implies(Bp(tmp_x1_ce_model, tmp_x2_ce_model) >= 0, Bp(tmp_x1_ce_model_next, tmp_x2_ce_model_next) >= 0))
        else:
            print("The non-negativity preservation property check for the certificate value across transitions has passed.")

        ce_solver.push()
        ce_solver.add(
                Q_Trans_Cond(x1_ce, x2_ce, qi, qj),
                In_X_Cond(x1_ce, x2_ce),
                In_X_Cond(x1_ce_next, x2_ce_next),
                Not_Acc_Cond(x1_ce, x2_ce, qi, qj),
                Bp_candidate(x1_ce, x2_ce) < Bp_candidate(x1_ce_next, x2_ce_next))
        if ce_solver.check() == sat:
            print("The non-increasing property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            tmp_x1_ce_model, tmp_x2_ce_model = t2float(ce_model[x1_ce]), t2float(ce_model[x2_ce])
            tmp_x1_ce_model_next,tmp_x2_ce_model_next =  f(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp = Bp_candidate2(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp2 = Bp_candidate2(tmp_x1_ce_model_next,tmp_x2_ce_model_next)
            print(f"x1_ce={tmp_x1_ce_model}, x2_ce={tmp_x2_ce_model}, Bp_c={tmp_bp}, Bp_c'={tmp_bp2}")
            s.add(Bp(tmp_x1_ce_model, tmp_x2_ce_model) >= Bp(tmp_x1_ce_model_next, tmp_x2_ce_model_next))
        else:
            print("The non-increasing property check for the certificate value across transitions has passed.")
        ce_solver.pop()
        ce_solver.push()
        ce_solver.add(
            Q_Trans_Cond(x1_ce, x2_ce, qi, qj),
            In_X_Cond(x1_ce, x2_ce),
            In_X_Cond(x1_ce_next, x2_ce_next),
            Not(Not_Acc_Cond(x1_ce, x2_ce, qi, qj)),
            Bp_candidate(x1_ce, x2_ce) < Bp_candidate(x1_ce_next, x2_ce_next) + candidate_model[epsilon])
        if ce_solver.check() == sat:
            print("The decreasing property of the certificate value across transitions is not satisfied.")
            ce_flag = True
            ce_model = ce_solver.model()
            tmp_x1_ce_model, tmp_x2_ce_model = t2float(ce_model[x1_ce]), t2float(ce_model[x2_ce])
            x1_ce_model_next, x2_ce_model_next = f(ce_model[x1_ce], ce_model[x2_ce])
            tmp_x1_ce_model_next, tmp_x2_ce_model_next = f(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp = Bp_candidate2(tmp_x1_ce_model, tmp_x2_ce_model)
            tmp_bp2 = Bp_candidate2(tmp_x1_ce_model_next, tmp_x2_ce_model_next) + t2float(candidate_model[epsilon])
            print(f"x1_ce={tmp_x1_ce_model}, x2_ce={tmp_x2_ce_model}\n"
                  f"x1_cen={tmp_x1_ce_model_next}, x2_cen={tmp_x2_ce_model_next}\n"
                  f"Bp_c={tmp_bp}, Bp_c'+e={tmp_bp2}, diff={tmp_bp-tmp_bp2}")
            s.add(Bp(ce_model[x1_ce], ce_model[x2_ce]) >= Bp(x1_ce_model_next, x2_ce_model_next) + epsilon)
        else:
            print("The decreasing property check for the certificate value across transitions has passed.")
        ce_solver.pop()
        if not ce_flag:
            cc_flag = True
            print("Certificate found!")
            break
        iter = iter + 1

    if iter > MAX_ITER:
        print("Exceeded the maximum number of iterations.")
    else:
        if not cc_flag:
            print("Unable to synthesize candidate certificate.")

start_time = time.time()
cegeis_synthesis(1, 1)
end_time = time.time()
print(f"Time cost: {end_time - start_time}")

"""
Experimental Results on March 15, 2025:
 
Certificate Template:
B11(x1, x2) = c0 + c1 * In_X0(x1, x2) + c2 * In_VF(x1, x2) + c3 * x1 * In_X0(x1, x2) + c4 * x2 * In_X0(x1, x2) + c5 * x1 * In_VF(x1, x2) + c6 * x2 * In_VF(x1, x2) + c7 * max(x1, x2) + c8 * x1 ** 2 + c9 * x2 ** 2
For synthesizing a persistent barrier certificate for (1, 1):
#1 Candidate Solution:
ε=0.25
c0=0.0
c1=0.0
c2=27.0
c3=0.0
c4=0.0
c5=-1.0
c6=0.0
c7=0.0
c8=0.0
c9=0.0
Initial non-negativity check passed
Transition non-negativity preservation property check passed
Non-increasing check passed
Non-negativity check passed
Descent check passed
Certificate found!
Time cost: 8.771111249923706
"""