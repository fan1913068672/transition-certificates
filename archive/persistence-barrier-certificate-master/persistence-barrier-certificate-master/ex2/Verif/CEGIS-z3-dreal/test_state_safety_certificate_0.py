import math
"""
Check if the state safety certificate using the CEGIS-z3-dreal method is valid.
"""
def In_X(x1, x2):
    return x1 >= 0 and x1 <= math.pi / 9 * 8 and x2 >= 0 and x2 <= math.pi / 9 * 8


def Bp10_2025_03_15(x1, x2, q):
    return -1 +  x1 * x2 / 8 + 3 / 2 * q ** 2 - 3 / 32 * q * x1 ** 2 - 153 / 512 * x2 * q

def f(x1, x2):
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega  + 1.69 + Ts * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega  + 1.69 + Ts * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p

def In_X0(x1, x2):
    return x1 >= 0 and x1 <= math.pi / 9 and x2 >= 0 and x2 <= math.pi / 9

def In_Unsafe(x1, x2):
    return x1 >= 5 / 6 * math.pi and x1 <= 8 / 9 * math.pi or x2 >= 5 / 6 * math.pi and x2 <= 8 / 9 * math.pi
def L(x1, x2):
    if In_Unsafe(x1, x2):
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

if __name__ == "__main__":
    a, b = 0, math.pi/9
    s = 0.01
    bad_flag = False
    MAX_IT = 2000
    samples = state_space_product(step_sample(a, b, s), step_sample(a, b, s))
    cegis_bf = Bp10_2025_03_15
    for x1, x2 in samples:
        logx1, logx2 = x1, x2
        q = 1
        bad_flag = False
        it = 0
        while it < MAX_IT:
            x1n, x2n = f(x1, x2)
            qn = delta(q, L(x1, x2))
            bf = cegis_bf(x1, x2, q)
            bfn = cegis_bf(x1n, x2n, qn)
            if In_X0(x1, x2) and bf < 0:
                bad_flag = True
                print("violate non-negativity property:", bf)
                break
            if In_X(x1, x2) and In_X(x1n, x2n):
                if bf >= 0 and bfn < 0:
                    bad_flag = True
                    print("violate non-negativity property through a transition:", x1, x2, q, bf, x1n, x2n, qn, bfn)
                    break
            if qn == 0 and bf >= 0:
                bad_flag = True
                print("violate reachability property:", bf, bfn)
                break
            if bf < 0:
                bad_flag = True
                print("violate non-negativity property")
                break
            x1, x2 = x1n, x2n
            q = qn
            it = it + 1
        print(f"{logx1}, {logx2} test passed")
    if bad_flag:
        print("There exists a counterexample")
    else:
        print("no counterexample")