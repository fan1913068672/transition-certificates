"""Check the transition persistence certificate synthesized by the CEGIS-z3-z3 method"""
def In_VF(x1_val, x2_val):
    return x1_val >= 20 and x1_val <= 26 and x2_val >= 20 and x2_val <= 26

def Bp11(x1, x2):
    return 28 * In_VF(x1, x2) - x2 * In_VF(x1, x2)

def Bp_2025_03_15(x1, x2):
    return 27 * In_VF(x1, x2) - x1 * In_VF(x1, x2)
ev_2025_03_15 = 0.25

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
def In_X0(x1, x2):
    return x1 >= 21 and x1 <= 24 and x2 >=21 and x2 <=24

def L(x1, x2):
    if In_VF(x1, x2) and In_X0(x1, x2):
        return 3
    if In_X0(x1, x2) and not In_VF(x1, x2):
        return 2
    if not In_X0(x1, x2) and In_VF(x1, x2):
        return 1
    if not In_X0(x1, x2) and not In_VF(x1, x2):
        return 0

def delta(q, w):
    if q == 0 and w >= 2:
        return 1
    if q == 0 and w < 2:
        return 2
    if q == 1:
        return 1
    if q == 2:
        return 2
def word(x1, x2):
    if L(x1,x2) in [3, 1]:
        return 'b'
    else:
        return "!b"

def step_sample(a, b, s):
    res = []
    for i in range(a * int(1/s), b*int(1/s)+1):
        res.append(i * s)

if __name__ == "__main__":
    a, b = 21, 24
    s = 0.05
    bad_flag = False
    MAX_IT = 1000
    epsilon11 = 0.25
    samples = []
    cegis_ev = ev_2025_03_15
    cegis_bf = Bp_2025_03_15
    for i in range(a * int(1/s), b*int(1/s)+1):
        x1 = i * s
        for j in range(a * int(1/s), b * int (1/s) + 1):
            x2 = j * s
            samples.append([x1, x2])
    for x1, x2 in samples:
        # print(x1, x2)
        logx1 = x1
        logx2 = x2
        q = 0
        bad_flag = False
        it = 0
        while it < MAX_IT:
            x1n, x2n = f(x1, x2)
            qn = delta(q, L(x1, x2))
            if q == 1 and qn == 1:
                bf = cegis_bf(x1, x2)
                bfn = cegis_bf(x1n, x2n)
                if In_VF(x1, x2):
                    if bf < bfn + cegis_ev:
                        bad_flag = True
                        print("Violation of the descent property:", bf, bfn + cegis_ev)
                        break
                else:
                    if bf < bfn:
                        bad_flag = True
                        print("Violation of the non-increasing property:", bf, bfn)
                        break
            else:
                bf = 0

            if bf < 0:
                bad_flag = True
                print("Violation of the non-negativity property:", logx1, logx2, bf)
                break
            x1, x2 = x1n, x2n
            q = qn
            it = it + 1
        print(f"{logx1}, {logx2} test passed")
    if bad_flag:
        print("Negative values have appeared in the certificate")
    else:
        print("No counterexamples have been detected")
