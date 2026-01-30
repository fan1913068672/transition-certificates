import math
"""
Check if the state safety certificate using the CEGIS-z3-dreal method is valid.
"""
def In_X(x):
    return x >= 0 and x <= math.pi * 2

def Bp10_2025_03_15(x, q):
    return -1 + 2 * q ** 2 - 7 / 16 * x * q

def f(x):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    xp = x + Ts * Omega + Ts * K * math.sin(-x) - 0.532 * x ** 2 + 1.69
    return xp

def In_X0(x):
    return x >= 0 and x <= math.pi / 9

def In_Unsafe(x):
    return x >= 7 / 9 * math.pi and x <= 8 / 9 * math.pi

def L(x):
    if In_Unsafe(x):
        return 1
    else:
        return 0

def delta(q, w):
    if q == 1 and w == 1:
        return 0
    if q == 1 and w == 0:
        return 1
    if q == 0:
        return 0

def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1/s)), int(b*int(1/s)+1)):
        res.append(i * s)
    return res

if __name__ == "__main__":
    a, b = 0, math.pi/9
    s = 0.0001
    bad_flag = False
    MAX_IT = 2000
    samples = step_sample(a, b, s)
    cegis_bf = Bp10_2025_03_15
    for x in samples:
        logx = x
        q = 1
        bad_flag = False
        it = 0
        while it < MAX_IT:
            xn = f(x)
            qn = delta(q, L(x))
            bf = cegis_bf(x, q)
            bfn = cegis_bf(xn, qn)
            if In_X0(x) and bf < 0:
                bad_flag = True
                print("violate non-negativity property:", logx, bf)
                break
            if In_X(x) and In_X(xn):
                if bf >= 0 and bfn < 0:
                    bad_flag = True
                    print("violate non-negativity property through a transition:", x, q, bf, xn, qn, bfn)
                    break
            if qn == 0 and bf >= 0:
                bad_flag = True
                print("violate reachability property:", bf, bfn)
                break
            if bf < 0:
                bad_flag = True
                print("violate non-negativity property")
                break
            x = xn
            q = qn
            it = it + 1
        print(f"{logx} test passed")
    if bad_flag:
        print("There exists a counterexample")
    else:
        print("no counterexample")