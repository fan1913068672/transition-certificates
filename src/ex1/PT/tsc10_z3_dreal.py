import math
import z3
import dreal
import time

"""
一维 Kuramoto 振子系统的安全性验证
LTL 规约: G !Xu (系统永远不会进入不安全状态)
对应的否定形式: F Xu
"""

# ==================== 系统参数定义 ====================
TS = 0.1          # 采样时间
OMEGA = 0.01      # 自然频率
K = 0.0006        # 耦合强度
PI = 3.1415926
# ==================== 集合定义 ====================
def In_X_Cond(x_ce):
    """定义状态空间 X 的条件: x ∈ [0, 2] (因为cos(0)=1)"""
    return dreal.And(x_ce >= 0, x_ce <= PI * 2)


def In_X0_Cond(x_ce):
    """定义初始状态集 X0 的条件: x ∈ [0, 1/9]"""
    return dreal.And(x_ce >= 0, x_ce <= PI / 9)


def In_Unsafe_Cond(x_ce):
    """定义不安全状态集 Xu 的条件: x ∈ [7π/9, 8π/9]"""
    return dreal.And(x_ce >= PI / 9 * 7, x_ce <= PI / 9 * 8)


def In_Unsafe(x):
    """检查数值x是否在不安全集中 (用于离散验证)"""
    return x >= 7 / 9 * PI and x <= 8 / 9 * PI

# ==================== 系统动力学 ====================
def f_t(x_ce):
    """
    连续时间动力学函数 (用于dreal验证)
    x_next = x + Ts*Omega + Ts*K*sin(-x) - 0.532*x^2 + 1.69
    """
    return x_ce + TS * OMEGA + TS * K * dreal.sin(-x_ce) - 0.532 * x_ce ** 2 + 1.69


def f_m(x):
    """
    离散时间动力学函数 (用于z3验证)
    与f_t相同，但使用math.sin而不是dreal.sin
    """
    xp = x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69
    return xp

# ==================== 模态自动机定义 ====================
def q_trans(q):
    """
    状态转移函数: 给定当前模态q，返回可能的下一模态列表
    q=1: 可转移到[0, 1]
    q=0: 吸收状态，只能保持在[0]
    """
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise Exception(f"无效的模态: {q}")


def delta(x, q):
    """
    模态转移的触发条件
    q=1: 如果x在不安全集中，转移到q=0；否则保持在q=1
    q=0: 吸收状态，保持在q=0
    """
    if q == 1:
        if In_Unsafe(x):
            return [0]      # 进入不安全状态，转移到吸收状态
        else:
            return [1]      # 保持安全状态
    else:
        return [0]          # 吸收状态，保持不变

# ==================== 辅助函数 ====================
def step_sample(a, b, s):
    """
    在区间[a, b]上以步长s生成采样点列表
    """
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res


def t2float(a, precision=14):
    """
    将z3的Real值转换为Python浮点数
    """
    s = a.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)


def space_product(s1, s2):
    """
    两个集合的笛卡尔积
    """
    def conn(x1, x2):
        if not isinstance(x1, list) and not isinstance(x2, list):
            return [x1, x2]
        elif isinstance(x1, list) and not isinstance(x2, list):
            return x1 + [x2]
        elif not isinstance(x1, list) and isinstance(x2, list):
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
    """
    多个集合的笛卡尔积
    """
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res

# ==================== 主要验证函数 ====================
def reachability_10():
    """
    使用Cegis框架合成状态安全证书
    验证系统从初始状态无法到达不安全状态
    """
    print("正在为状态0合成安全证书...")
    cc_flag = False
    
    # 定义证书系数 (9个参数)
    c = [z3.Real(f'c{i}') for i in range(0, 9)]
    s = z3.SolverFor("QF_NRA")
    
    # 生成采样点
    X_Samples = step_sample(0, 2 * PI, 0.01)  # 状态空间采样
    Q_Samples = [0, 1]                             # 模态空间
    Y_Samples = state_space_product(X_Samples, Q_Samples)  # 组合状态空间
    
    X0_Samples = step_sample(0, PI / 9, 0.01)  # 初始状态采样
    Q0_Samples = [1]                                # 初始模态
    Y0_Samples = state_space_product(X0_Samples, Q0_Samples)  # 初始组合状态
    
    Qacc_Samples = [0]                              # 吸收模态
    Yu_Samples = state_space_product(X_Samples, Qacc_Samples)  # 不安全状态
    
    
    print(f"  D_0: 初始状态对数量 = {len(Y0_Samples)}")
    print(f"  D_u: 不安全状态对数量 = {len(Yu_Samples)}")
    print(f"  D_x: 完整状态空间大小 = {len(Y_Samples)}")
    def Bp_t(x, q):
        """
        安全证书模板: 4次多项式
        B(x, q) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*q + c6*q^2 + c7*q^3 + c8*x*q
        """
        res = (c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4 +
               c[5] * q + c[6] * q ** 2 + c[7] * q ** 3 + c[8] * x * q)
        return z3.simplify(res)
    
    # 约束1: 初始状态B(x,q) ≥ 0
    for x0, q0 in Y0_Samples:
        s.add(Bp_t(x0, q0) >= 0)
    
    # 约束2: 不安全状态B(x,q) < 0
    for x, qacc in Yu_Samples:
        s.add(Bp_t(x, qacc) < 0)
    
    # 约束3: 前向不变性: 如果B(x,q) ≥ 0，则B(x',q') ≥ 0
    for x, q in Y_Samples:
        xp = f_m(x)  # 下一状态
        qp_list = delta(x, q)  # 可能的下一模态
        for qp in qp_list:
            s.add(z3.Implies(Bp_t(x, q) >= 0, Bp_t(xp, qp) >= 0))
    
    # 设置Cegis循环
    MAX_ITER = 1000
    iter = 0
    
    # 创建dreal求解器用于反例生成
    ce_solver = dreal.Context()
    ce_solver.config.precision = 0.0001
    ce_solver.SetLogic(dreal.Logic.QF_NRA)
    x_ce = dreal.Variable('x_ce')
    ce_solver.DeclareVariable(x_ce, 0, 2 * dreal.cos(0))
    
    # Cegis主循环
    while s.check() == z3.sat and iter < MAX_ITER:
        ce_flag = False
        m = s.model()
        c_m = [t2float(m[i]) for i in c]
        
        print(f"迭代 #{iter}:")
        for i, cc in enumerate([m[item] for item in c]):
            print(f"c{i}={t2float(cc)}")
        
        def Bp_c(x, q):
            """使用当前系数评估证书函数"""
            return (c_m[0] + c_m[1] * x + c_m[2] * x ** 2 + c_m[3] * x ** 3 + c_m[4] * x ** 4 +
                    c_m[5] * q + c_m[6] * q ** 2 + c_m[7] * q ** 3 + c_m[8] * x * q)
        
        # 反例检查1: 初始状态非负性
        ce_solver.Push(2)
        ce_solver.Assert(In_X0_Cond(x_ce))
        ce_solver.Assert(Bp_c(x_ce, 1) < 0)  # 寻找违反B≥0的初始状态
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("发现初始状态非负性的反例")
            print(f"x={ce_model[x_ce].mid()}, B={Bp_c(ce_model[x_ce].mid(), 1)}")
            ce_flag = True
            s.add(Bp_t(ce_model[x_ce].mid(), 1) >= 0)  # 添加新约束
        else:
            print("初始状态非负性验证通过")
        ce_solver.Pop(2)
        
        # 反例检查2: 不安全状态负性
        ce_solver.Push(2)
        ce_solver.Assert(In_X_Cond(x_ce))
        ce_solver.Assert(Bp_c(x_ce, 0) >= 0)  # 寻找违反B<0的不安全状态
        ce_model = ce_solver.CheckSat()
        if ce_model is not None:
            print("发现不安全状态负性的反例")
            print(f"x={ce_model[x_ce].mid()}, B={Bp_c(ce_model[x_ce].mid(), 0)}")
            ce_flag = True
            s.add(Bp_t(ce_model[x_ce].mid(), 0) < 0)  # 添加新约束
        else:
            print("不安全状态负性验证通过")
        ce_solver.Pop(2)
        
        # 反例检查3: 模态q=1的前向不变性
        tnn_flag = False
        ce_solver.Push(3)
        ce_solver.Assert(In_X_Cond(x_ce))
        ce_solver.Assert(In_X_Cond(f_t(x_ce)))
        ce_solver.Assert(Bp_c(x_ce, 1) >= 0)
        qp_list = q_trans(1)
        
        for qp in qp_list:
            ce_solver.Push(2)
            if qp == 0:  # 转移到吸收状态
                ce_solver.Assert(In_Unsafe_Cond(x_ce))  # 必须在不安全集中
                ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)  # 下一状态B<0
            elif qp == 1:  # 保持在安全状态
                ce_solver.Assert(dreal.Not(In_Unsafe_Cond(x_ce)))  # 不在不安全集中
                ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)  # 下一状态B<0
            else:
                raise Exception(f"无效的模态: {qp}")
            
            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print("发现前向不变性的反例 (q=1)")
                ce_flag = True
                tnn_flag = True
                x_ce_m = ce_model[x_ce].mid()
                x_ce_p_m = f_m(x_ce_m)
                print(f"B(x,q)={Bp_c(x_ce_m, 1)}, B(x',q')={Bp_c(x_ce_p_m, qp)}")
                s.add(z3.Implies(Bp_t(x_ce_m, 1) >= 0, Bp_t(x_ce_p_m, qp) >= 0))
            ce_solver.Pop(2)
        ce_solver.Pop(3)
        
        # 反例检查4: 模态q=0的前向不变性
        ce_solver.Push(3)
        ce_solver.Assert(In_X_Cond(x_ce))
        ce_solver.Assert(In_X_Cond(f_t(x_ce)))
        ce_solver.Assert(Bp_c(x_ce, 0) >= 0)
        qp_list = q_trans(0)
        
        for qp in qp_list:
            ce_solver.Push(1)
            if qp == 0:  # 吸收状态
                ce_solver.Assert(Bp_c(f_t(x_ce), qp) < 0)  # 下一状态B<0
            else:
                raise Exception(f"无效的模态: {qp}")
            
            ce_model = ce_solver.CheckSat()
            if ce_model is not None:
                print("发现前向不变性的反例 (q=0)")
                ce_flag = True
                tnn_flag = True
                x_ce_m = ce_model[x_ce].mid()
                x_ce_p_m = f_m(x_ce_m)
                print(f"B(x,q)={Bp_c(x_ce_m, 0)}, B(x',q')={Bp_c(x_ce_p_m, qp)}")
                s.add(z3.Implies(Bp_t(x_ce_m, 0) >= 0, Bp_t(x_ce_p_m, qp) >= 0))
            ce_solver.Pop(1)
        ce_solver.Pop(3)
        
        if not tnn_flag:
            print("前向不变性验证通过")
        
        # 如果没有找到反例，则找到有效证书
        if not ce_flag:
            print("成功合成安全证书：")
            cc_flag = True
            for i, cc in enumerate([m[item] for item in c]):
                print(f"c{i}={cc}")
            break
        
        iter += 1
    
    # 输出结果
    if iter >= MAX_ITER:
        print("超过最大迭代次数")
    elif not cc_flag:
        print("无法合成状态安全证书")
    else:
        print(f"总共迭代次数: {iter}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    start_time = time.time()
    reachability_10()
    end_time = time.time()
    print(f"安全证书合成用时: {end_time - start_time:.4f} 秒")

"""
目标: 验证系统无法从模态q=1转移到q=0 (即无法到达不安全状态)

2025/03/15 实验结果:
证书模板: B(x, q) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*q + c6*q^2 + c7*q^3 + c8*x*q
证书系数:
c0 = -1
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 2
c7 = 0
c8 = -7/16
安全证书合成用时: 4.2516 秒
"""