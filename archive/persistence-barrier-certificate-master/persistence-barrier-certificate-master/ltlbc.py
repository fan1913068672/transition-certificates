from z3 import *


# # 定义状态变量（假设q为整数）
# x1 = Real('x1')
# x2 = Real('x2')
# q = Int('q')
#
# # 定义B(y)的多项式表达式
# B_y = c0 + c1*x1 + c2*x2 + c3 * ToReal(q) + c4*x1**2 + c5*x2**2 + c6*(ToReal(q)**2)

# 离散时间系统的状态方程
def f(x1, x2, q):
    alpha = 0.004
    theta = 0.01
    Te = 0
    Th = 40
    mu = 0.15
    def u(x_curr):
        return 0.59 - 0.011 * x_curr
    x1_next = (1-2*alpha-theta-mu*u(x1)) * x1 + x2 * alpha + mu*Th*u(x1) + theta * Te
    x2_next = x1 * alpha + (1-2*alpha-theta-mu*u(x2)) * x2 + mu*Th*u(x2) + theta * Te
    q_next_list = delta(q, L(x1, x2))
    return state_space_product([x1_next], [x2_next], q_next_list)

# 标签函数L(x)（示例：根据状态区域生成标签）
def L(x1_val, x2_val):
    cond1 = x1_val >= 21 and x1_val <= 24 and x2_val >= 21 and x2_val <= 24
    cond2 = x1_val >= 20 and x1_val <= 26 and x2_val >= 20 and x2_val <= 26
    if cond1 and cond2:
        return 3
    if cond1 and not cond2:
        return 2
    if not cond1 and cond2:
        return 1
    if not cond1 and not cond2:
        return 0
# 自动机迁移函数δ(q, a)
def delta(q_current, label):
    if q_current == 0:
        if label >= 2: # only a
            return [1, 2]
        else:
            return []
    elif q_current == 1:
        return [1]
    elif q_current == 2:
        return []

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

def InVF(x1, x2, q):
    return q == 1

def InY0(x1, x2, q):
    return x1 >= 21 and x1 <= 24 and x2 >= 21 and x2 <= 24 and q == 0

def InY(x1, x2, q):
    return x1 >= 20 and x1 <= 34 and x2 >= 20 and x2 <= 34 and (q == 0 || q == 1 || q == 2)


def cegeis_synthesis():
    # 定义系数和ε变量
    c0 = Real('c0')
    c1 = Real('c1')
    c2 = Real('c2')
    c3 = Real('c3')
    c4 = Real('c4')
    c5 = Real('c5')
    c6 = Real('c6')
    epsilon = Real('epsilon')
    # 定义求解器
    s = Solver()
    samples = []  # 存储反例样本
    max_iterations = 20
    # 条件1的采样: Bp(y0) > 0
    C1_X1_Samples = list(range(21, 25))
    C1_X2_Samples = list(range(21, 25))
    C1_Q_Samples = [0,]
    C1_Samples = state_space_product(C1_X1_Samples, C1_X2_Samples, C1_Q_Samples)
    # 条件2的采样: Bp(y) >= Bp(y') + I(y')e
    C2_X1_Samples = list(range(20, 35))
    C2_X2_Samples = list(range(20, 35))
    C2_Q_Samples = [0, 1, 2]
    C2_Samples = state_space_product(C2_X1_Samples, C2_X2_Samples, C2_Q_Samples)
    # 条件3的采样: Bp(y) > 0 -> Bp(y') > 0
    C3_X1_samples = list(range(20, 35))
    C3_X2_samples = list(range(20, 35))
    C3_Q_Samples = [0, 1, 2]
    C3_Samples = state_space_product(C3_X1_samples, C3_X2_samples, C3_Q_Samples)
    # TODO：将x1, x2, q实例化，不然合成对象他搞不清楚是什么，把q的转移过程编码成条件
    def Bp(x1, x2, q):
        return c0 + c1 * x1 + c2 * x2 + c3 * ToReal(q) + c4 * x1 ** 2 + \
               c5 * x2 ** 2 + c6 * (ToReal(q) ** 2)

    # 将条件1相关采样的条件添加到求解器中: Bp(y0) > 0
    for x1, x2, q in C1_Samples:
        s.add(Bp(x1, x2, q) > 0)
    # 将条件2相关采样的条件添加到求解器中: Bp(y) >= Bp(y') + I(y')e
    for x1, x2, q in C2_Samples:
        # 根据y'所在区域构建不同的条件
        for x1_next, x2_next, q_next in f(x1, x2, q):
            if InVF(x1_next, x2_next, q_next):
                s.add(Bp(x1, x2, q) >= Bp(x1_next, x2_next, q_next) + e)
            else:
                s.add(Bp(x1, x2, q) >= Bp(x1_next, x2_next, q_next))
    # 将条件2相关采样的条件添加到求解器中: Bp(y) > 0 ==> Bp(y') > 0
    for x1, x2, q in C3_Samples:
        for x1_next, x2_next, q_next in f(x1, x2, q):
            s.add(Implies(Bp(x1, x2, q) > 0, Bp(x1_next, x2_next, q_next) > 0))
    # 初始约束：ε > 0
    s.add(epsilon > 0)

    for _ in range(max_iterations):
        # 1. 合成阶段：求解当前约束下的系数
        if s.check() == sat:
            model = s.model()
            print(f"候选解: ε={model[epsilon]}, c={[model[c] for c in [c0, c1, c2, c3, c4, c5, c6]]}")

            # 2. 验证阶段：检查是否存在反例
            violation = find_violation(model)
            if not violation:
                print("找到可行解!")
                return model
            else:
                # 3. 添加反例约束
                add_counterexample_constraint(s, violation)
                samples.append(violation)
        else:
            print("无解")
            return None


def find_violation(model):
    # 检查反例
    solver = Solver()
    # 提取当前候选证书的系数和下降常数ε
    c0_val = model[c0]
    c1_val = model[c1]
    c2_val = model[c2]
    c3_val = model[c3]
    c4_val = model[c4]
    c5_val = model[c5]
    c6_val = model[c6]
    epsilon_val = model[epsilon]

    # 定义B(y)的表达式
    B_y_val = c0_val + c1_val * x1 + c2_val * x2 + c3_val * ToReal(q) + c4_val * x1 ** 2 + \
              c5_val * x2 ** 2 + c6_val * (ToReal(q) ** 2)
    # 计算F(y)的下一个状态
    x1_next, x2_next = f(x1, x2)
    label = L(x1, x2)
    q_next = delta(q, label)
    B_y_next_val = c0_val + c1_val * x1_next + c2_val * x2_next + c3_val * ToReal(
        q_next) + c4_val * x1_next ** 2 + c5_val * x2_next ** 2 + c6_val * (ToReal(q_next) ** 2)

    # 检查约束违反情况
    # 条件1: 如果y0 ∈ Y0, B(y0) > 0
    cond1 = Implies(InY0(x1, x2, q), B_y_val > 0)
    # 条件2: 如果F(y) ∈ VF，则 B(y) ≥ B(F(y)) + ε
    cond2 = Implies(InVF(x1, x2, q_next), B_y_val >= B_y_next_val + epsilon_val)
    # 条件3: 如果F(y) ∉ VF，则 B(y) ≥ B(F(y))
    cond3 = Implies(Not(InVF(x1, x2, q_next)), B_y_val >= B_y_next_val)
    # 条件4: 如果y 属于 Y， 则如果 B(y) > 0 则B(y') > 0
    cond4 = Implies(InY(x1, x2, q), Implies(B_y_val > 0, B_y_next_val > 0))
    solver.push()
    solver.add(Or(Not(cond1), Not(cond2), Not(cond3), Not(cond4)))
    if solver.check() == sat:
        return solver.model()
    else:
        return None

def add_counterexample_constraint(s, violation):
    # 提取反例状态值
    x1_cex = violation[x1]
    x2_cex = violation[x2]
    q_cex = violation[q]

    # 计算F(y)的下一状态
    x1_next, x2_next = f(x1_cex, x2_cex)
    label = L(x1_cex, x2_cex)
    q_next = delta(q_cex, label)
    B_y_next = c0 + c1 * x1_next + c2 * x2_next + c3 * ToReal(q_next) + c4 * x1_next ** 2 + \
               c5 * x2_next ** 2 + c6 * (ToReal(q_next) ** 2)
    # 添加约束：避免此反例
    s.add(Not( # 不要发生反例这种情况加到合成候选证书的约束中
        And( # 反例情况的描述
    x1 == x1_cex,
        x2 == x2_cex,
        q == q_cex,
        Or(
            B_y <= 0,
            And(Not(InVF(q_next)), B_y < B_y_next),
            And(InVF(q_next), B_y < B_y_next + epsilon)
        ))))

