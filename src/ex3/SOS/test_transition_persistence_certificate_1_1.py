"""Check the certificate synthesized by SOS"""
def In_VF(x1_val, x2_val):
    return x1_val >= 20 and x1_val <= 26 and x2_val >= 20 and x2_val <= 26

def Bp11(x1, x2):
    return 2.855 * (-0.003 * x1 - 0.013 * x2 + 1) ** 2 + 1.113 * (-0.999 * x1 + x2) ** 2

def Bp11_2025_03_16_03_11(x1, x2):
    return 4.064 * (-0.004 * x1 - 0.005 * x2 + 1) ** 2 + 1.092 * (-1.0 * x1 + x2) ** 2+ 0.001 * x1 ** 2
epsilon11_2025_03_16_03_12 = 1

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
    a, b = 20, 24
    s = 0.05
    bad_flag = False
    MAX_IT = 1000
    sosev = epsilon11_2025_03_16_03_12
    sosbf = Bp11_2025_03_16_03_11
    samples = []
    for i in range(a * int(1/s), b*int(1/s)+1):
        x1 = i * s
        for j in range(a * int(1/s), b * int (1/s) + 1):
            x2 = j * s
            samples.append([x1, x2])

    # Perform a 1000-step check starting from each sampling point
    for x1, x2 in samples:
        logx1 = x1
        logx2 = x2
        q = 0
        bad_flag = False
        it = 0
        while it < MAX_IT:
            x1n, x2n = f(x1, x2)
            qn = delta(q, L(x1, x2))
            if q == 1 and qn == 1:
                bf = sosbf(x1, x2)
                if In_VF(x1, x2):
                    if bf < Bp11(x1n, x2n) + sosev:
                        bad_flag = True
                        print("Violation of the descent property:", bf, sosbf(x1n, x2n) + sosev)
                        break
                else:
                    if bf < Bp11(x1n, x2n):
                        bad_flag = True
                        print("Violation of non-increasing property:", bf, sosbf(x1n, x2n))
                        break
            else:
                bf = 0

            if bf < 0:
                bad_flag = True
                print("Violation of non-negativity property:", logx1, logx2, bf)
                break
            x1, x2 = x1n, x2n
            q = qn
            it = it + 1
        print(f"{logx1}, {logx2} test passed")
    if bad_flag:
        print("Negative values appeared in the certificate.")
    else:
        print("No counterexamples were found.")

