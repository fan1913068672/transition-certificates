ce1 = (21.0, 21.0)


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

def In_X0(x1, x2):
    return x1 >= 21 and x1 <= 24 and x2 >=21 and x2 <=24

def check(x1, x2, i):
    # 偶数状态出现在X0中
    if i % 2 != 0:
        return True
    else:
        if In_X0(x1, x2):
            return True
        else:
            return False

def main(x1, x2):
    i = 1
    flag = True
    while True:
        print(x1, x2, i)
        if not check(x1, x2, i):
            break
        x1, x2 = f_m(x1, x2)
        i = i + 1
    if not flag:
        print("violate specification")


main(*ce1)