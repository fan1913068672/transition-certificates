from z3 import *


def synthesize_bf1():
    # 定义待合成的系数和常量
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = Reals('c0 c1 c2 c3 c4 c5 c6 c7 c8')
    e = Real('e')
    s = Solver()

    # 定义证书函数 bf(x) = c0 + c1*x + c2*x² + c3*x³
    def bf(x):
        return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x**4 + c5 * x**5 + c6 * x**6+ c7 * x**7+ c8 * x**8

    # 定义函数 f(x) = x - 0.1 表示离散时间系统方程
    def f(x):
        return x - 0.1

    # ---------------------------
    # 阶段1：基于采样数据构建约束
    # ---------------------------

    # 约束1：当 f(x) ∈ VF={-3, -2, -1} 时，bf(f(x)) + e <= bf(x)
    # 对应 x ∈ {-6, -5, -4}（因为 f(x) = x+3 ∈ {0,1,2}）
    S3 = [0.1, 0.2, 0.5, 1.1]
    for x in S3:
        s.add(bf(f(x)) + e <= bf(x))

    # 约束2：当 f(x) ∉ VF 时，bf(f(x)) <= bf(x)
    # 采样点选择 x ∈ {0, 1, 2}（对应 f(x) = 3,4,5 ∉ VF）
    S2 = [-0.3, 1.2, 2.5, -4.9]
    for x in S2:
        s.add(bf(f(x)) <= bf(x))

    # 约束3：随机采样点保证 bf(x) > 0
    # 选择 x ∈ {1, 2, 3}（x > 0 的初始状态）
    S1 = [0.2, -0.2, 0.3, -0.5]
    for x in S1:
        s.add(bf(x) > 0)

    # 约束4：e 必须是正数
    s.add(e > 0)

    # ---------------------------
    # 求解约束
    # ---------------------------
    while s.check() == sat:
        print("--------------------------------------")
        flag = False # 表示没有找到反例
        model = s.model()
        print("候选证书找到，参数为：")
        print(f"c0 = {model[c0]}")
        print(f"c1 = {model[c1]}")
        print(f"c2 = {model[c2]}")
        print(f"c3 = {model[c3]}")
        print(f"c4 = {model[c4]}")
        print(f"c5 = {model[c5]}")
        print(f"c6 = {model[c6]}")
        print(f"c7 = {model[c7]}")
        print(f"c8 = {model[c8]}")
        print(f"e = {model[e]}")

        # ---------------------------
        # 阶段2：验证反例
        # ---------------------------
        verify = Solver()

        # 使用合成后的参数定义新约束
        c0_val = model[c0]
        c1_val = model[c1]
        c2_val = model[c2]
        c3_val = model[c3]
        c4_val = model[c4]
        c5_val = model[c5]
        c6_val = model[c6]
        c7_val = model[c7]
        c8_val = model[c8]
        e_val = model[e]

        def bf_verify(x):
            return c0_val + c1_val * x + c2_val * x ** 2 + c3_val * x ** 3 + c4_val * x ** 4 + c5_val * x ** 5 + c6_val * x ** 6 + c7_val * x ** 7 + c8_val * x ** 8

        # 验证条件1：当 x > 0 初始状态集合 时 bf(x) > 0
        x_all = Real('x_all')
        verify.push()
        verify.add(x_all>=-10, x_all <= 10, bf_verify(x_all) <= 0)
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            # 优先显示具体数值反例
            if m[x_all].as_decimal(3) is not None:
                print(f"x = {m[x_all]} 时违反 bf(x) > 0")
            s.add(bf(m[x_all]) > 0)
        verify.pop()


        # 验证条件2：当 f(x) < 0 即f(x)属于VF 时 bf(f(x)) + e <= bf(x)
        # 对应 x < -3（因为 f(x) = x+3 < 0 → x < -3）
        x_vf = Real('x_vf')
        verify.push()
        verify.add(f(x_vf) >= 0, f(x_vf) <= 1, x_vf >= -10, x_vf <= 10,
                   bf_verify(f(x_vf)) + e_val > bf_verify(x_vf))
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            s.add(bf(f(m[x_vf])) + e <= bf(m[x_vf]))
            if m[x_vf].as_decimal(3) is not None:
                print(f"x = {m[x_vf]} 时违反 bf(f(x)) + e <= bf(x)")
        verify.pop()

        # 验证条件3：当 f(x) >= 0 时 bf(f(x)) <= bf(x)
        x_non_vf1 = Real('x_non_vf1')
        verify.push()
        verify.add(f(x_vf) < 0, f(x_vf) >= -10, x_vf >= -10, x_vf <= 10,
                   bf_verify(f(x_non_vf1)) > bf_verify(x_non_vf1))
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            s.add(bf(f(m[x_non_vf1])) <= bf(m[x_non_vf1]))
            if m[x_non_vf1].as_decimal(3) is not None:
                print(f"x = {m[x_non_vf1]} 时违反 bf(f(x)) <= bf(x)")
        verify.pop()

        x_non_vf2 = Real('x_non_vf2')
        verify.push()
        verify.add(f(x_vf) > 1, f(x_vf) <= 10, x_vf >= -10, x_vf <= 10,
                   bf_verify(f(x_non_vf2)) > bf_verify(x_non_vf2))
        # 检查是否存在反例
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            s.add(bf(f(m[x_non_vf2])) <= bf(m[x_non_vf2]))
            if m[x_non_vf2].as_decimal(3) is not None:
                print(f"x = {m[x_non_vf2]} 时违反 bf(f(x)) <= bf(x)")
        verify.pop()
        if not flag:
            print("合成成功")
            return model
        else:
            print("存在反例")
    else:
        print("无法找到满足条件的解")
        return None

def synthesize_bf2():
    # 定义待合成的系数和常量
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = Reals('c0 c1 c2 c3 c4 c5 c6 c7 c8')
    e = Real('e')
    s = Solver()

    # 定义证书函数 bf(x) = c0 + c1*x + c2*x² + c3*x³
    def bf(x):
        return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x**4 + c5 * x**5 + c6 * x**6+ c7 * x**7+ c8 * x**8

    # 定义函数 f(x) = x - 0.1 表示离散时间系统方程
    def f(x):
        return x + 0.1

    S3 = [0.9, 2.0, 3.0] # vf : x >= 1.0
    for x in S3:
        s.add(bf(f(x)) + e <= bf(x))

    S2 = [-10, -9, -8]
    for x in S2:
        s.add(bf(f(x)) <= bf(x))

    # 约束3：随机采样点保证 bf(x) > 0
    # 选择 x ∈ {1, 2, 3}（x > 0 的初始状态）
    S1 = [0.2, -0.2, 0.3, -0.5]
    for x in S1:
        s.add(bf(x) > 0)

    # 约束4：e 必须是正数
    s.add(e > 0)

    # ---------------------------
    # 求解约束
    # ---------------------------
    MAX_TIME = 500
    iter = 0
    while s.check() == sat and iter <= MAX_TIME:
        iter = iter + 1
        print("--------------------------------------")
        flag = False # 表示没有找到反例
        model = s.model()
        print("候选证书找到，参数为：")
        print(f"c0 = {model[c0]}")
        print(f"c1 = {model[c1]}")
        print(f"c2 = {model[c2]}")
        print(f"c3 = {model[c3]}")
        print(f"c4 = {model[c4]}")
        print(f"c5 = {model[c5]}")
        print(f"c6 = {model[c6]}")
        print(f"c7 = {model[c7]}")
        print(f"c8 = {model[c8]}")
        print(f"e = {model[e]}")

        # ---------------------------
        # 阶段2：验证反例
        # ---------------------------
        verify = Solver()

        # 使用合成后的参数定义新约束
        c0_val = model[c0]
        c1_val = model[c1]
        c2_val = model[c2]
        c3_val = model[c3]
        c4_val = model[c4]
        c5_val = model[c5]
        c6_val = model[c6]
        c7_val = model[c7]
        c8_val = model[c8]
        e_val = model[e]

        def bf_verify(x):
            return c0_val + c1_val * x + c2_val * x ** 2 + c3_val * x ** 3 + c4_val * x ** 4 + c5_val * x ** 5 + c6_val * x ** 6 + c7_val * x ** 7 + c8_val * x ** 8

        # 验证条件1：当 x > 0 初始状态集合 时 bf(x) > 0
        x_all = Real('x_all')
        verify.push()
        verify.add(x_all>=-10, bf_verify(x_all) <= 0)
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            # 优先显示具体数值反例
            if m[x_all].as_decimal(3) is not None:
                print(f"x = {m[x_all]} 时违反 bf(x) > 0")
            s.add(bf(m[x_all]) > 0)
        verify.pop()


        # 验证条件2：当 f(x) < 0 即f(x)属于VF 时 bf(f(x)) + e <= bf(x)
        # 对应 x < -3（因为 f(x) = x+3 < 0 → x < -3）
        x_vf = Real('x_vf')
        verify.push()
        verify.add(f(x_vf) >= 1, x_vf >= -10,
                   bf_verify(f(x_vf)) + e_val > bf_verify(x_vf))
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            s.add(bf(f(m[x_vf])) + e <= bf(m[x_vf]))
            if m[x_vf].as_decimal(3) is not None:
                print(f"x = {m[x_vf]} 时违反 bf(f(x)) + e <= bf(x)")
        verify.pop()

        # 验证条件3：当 f(x) >= 0 时 bf(f(x)) <= bf(x)
        x_non_vf1 = Real('x_non_vf1')
        verify.push()
        verify.add(f(x_vf) < 1, x_vf >= -10,
                   bf_verify(f(x_non_vf1)) > bf_verify(x_non_vf1))
        if verify.check() == sat:
            flag = True
            print("\n发现反例：")
            m = verify.model()
            s.add(bf(f(m[x_non_vf1])) <= bf(m[x_non_vf1]))
            if m[x_non_vf1].as_decimal(3) is not None:
                print(f"x = {m[x_non_vf1]} 时违反 bf(f(x)) <= bf(x)")
        verify.pop()

        if not flag:
            print("合成成功")
            return model
        else:
            print("存在反例")
    if iter > MAX_TIME:
        print("已达到最大迭代次数")
    else:
        print("无法找到候选证书")
    return None


# 执行合成与验证
model = synthesize_bf2()