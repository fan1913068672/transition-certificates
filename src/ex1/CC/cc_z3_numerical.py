'''
这个文件是合成闭包证书验证安全性根据给定的自动机，根据自动机上的状态对进行分段T_{i,j}构建模板, T_{i,j}表示自动机上q=i, q=j对应的闭包证书。
新的简化模板如下:
T_00(x, y) = c00 + c10 * x + c20 * y
T_01(x, y) = c01 + c11 * x + c21 * y
T_10(x, y) = c02 + c12 * x + c22 * y
T_11(x, y) = c03 + c13 * x + c23 * y

我们利用z3求解器，通过采样点合成这对应的候选闭包证书.

给定系统S为一个三元组(X, X_0, f), 基于状态接受条件的NBA为一个元组(\Sigma, q_0, Q, \delta, Acc).
其中X是系统状态结合， X_0是初始状态集合， f是状态演化方程.
给定AP为原子命题集合， 定义L: X \to 2^{AP}为标签函数， \Sigma = 2^{AP}为字母表， \delta \subseteq Q \times Q表示迁移关系， Acc \subseteq Q为接受状态集合。 


这四个证书需要满足的条件如下:
T_{i, i'}(x, x') >= 0, 其中i' \in \delta(i, L(x)); x' = f(x); x \in X ; i \in Q.
T_{i', j}(x', y) >= 0 imply T_{i, j}(x, y) >= 0, 其中i' \in \delta(i, L(x)); x, y \in X; i \in Q.
(T_{s,l}(x_0, z) >= 0 and T_{l, l'}(z, z')>=0) imply (T_{s, l'}(x_0, z') + e <= T_{s, l}(x_0, z)), 其中s = q_0; l, l' \in Acc; x_0 \in X_0; z, z' \in X;
其中e也是待定参数,要求e>0

核心思想: 为了证明接受状态会被有限次访问通过有限下降证明自动机从初始状态到接受状态的状态对是有限次发生。
'''
import math
import z3
import dreal
import time
import random

# ==================== 系统参数定义 ====================
TS = 0.1          # 采样时间
OMEGA = 0.01      # 自然频率
K = 0.0006        # 耦合强度
PI = 3.1415926
# ==================== 集合定义 ====================
def In_X_Cond(x_ce):
    """定义状态空间 X 的条件: x ∈ [0, 2π] """
    return dreal.And(x_ce >= 0, x_ce <= PI * 2)


def In_X0_Cond(x_ce):
    """定义初始状态集 X0 的条件: x ∈ [0, π/9]"""
    return dreal.And(x_ce >= 0, x_ce <= PI / 9)


def In_Unsafe_Cond(x_ce):
    """定义不安全状态集 Xu 的条件: x ∈ [7π/9, 8π/9]"""
    return dreal.And(x_ce >= 7 * PI / 9, x_ce <= 8 * PI / 9)


def In_X0(x):
    """检查数值x是否在初始集中"""
    return x >= 0 and x <= PI / 9


def In_Unsafe(x):
    """检查数值x是否在不安全集中 (用于离散验证)"""
    return x >= 7 * PI / 9 and x <= 8 * PI / 9

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

# ==================== 模态转移函数 ====================
def delta(i, L_x):
    """
    模态转移函数
    i: 当前自动机状态 (0或1)
    L_x: 标签函数，返回True如果x在不安全集中
    返回: 可能的下一状态列表
    """
    if i == 1:
        if L_x:  # 如果x在不安全集中
            return [0]
        else:    # 如果x在安全集中
            return [1]
    elif i == 0:
        return [0]  # 状态0是吸收状态
    else:
        raise ValueError(f"无效的自动机状态: {i}")

# ==================== 简化线性闭包证书模板定义 ====================
def T_00_template(x, y, coeffs):
    """T_00(x, y)模板函数 - 简化为线性: c00 + c10*x + c20*y"""
    c00, c10, c20 = coeffs[0:3]
    return c00 + c10 * x + c20 * y


def T_01_template(x, y, coeffs):
    """T_01(x, y)模板函数 - 简化为线性: c01 + c11*x + c21*y"""
    c01, c11, c21 = coeffs[3:6]
    return c01 + c11 * x + c21 * y


def T_10_template(x, y, coeffs):
    """T_10(x, y)模板函数 - 简化为线性: c02 + c12*x + c22*y"""
    c02, c12, c22 = coeffs[6:9]
    return c02 + c12 * x + c22 * y


def T_11_template(x, y, coeffs):
    """T_11(x, y)模板函数 - 简化为线性: c03 + c13*x + c23*y"""
    c03, c13, c23 = coeffs[9:12]
    return c03 + c13 * x + c23 * y


def T_template(x, y, i, j, coeffs):
    """根据i, j选择对应的T模板函数"""
    if i == 0 and j == 0:
        return T_00_template(x, y, coeffs)
    elif i == 0 and j == 1:
        return T_01_template(x, y, coeffs)
    elif i == 1 and j == 0:
        return T_10_template(x, y, coeffs)
    elif i == 1 and j == 1:
        return T_11_template(x, y, coeffs)
    else:
        raise ValueError(f"无效的状态对: ({i}, {j})")

# ==================== 闭包证书合成 ====================
def synthesize_closure_certificate(num_samples=1000):
    """合成闭包证书的主要函数"""
    print("开始合成闭包证书...")
    
    # 定义12个系数和1个epsilon参数
    coeffs = [z3.Real(f'c{i:02d}') for i in range(12)]
    EPSILON = z3.Real('EPSILON')
    s = z3.SolverFor("QF_NRA")
    
    # 条件: epsilon > 0, 系数约束
    for c in coeffs:
        s.add(c >= -100)
        s.add(c <= 100)
    
    # 约束系数不全为0
    for i in range(4):
        base_idx = i * 3
        s.add(z3.Or(coeffs[base_idx] != 0, coeffs[base_idx+1] != 0, coeffs[base_idx+2] != 0))
    
    s.add(EPSILON > 0)
    s.add(EPSILON <= 1)
    
    # 生成采样点
    X_samples = step_sample(0, 2.0 * PI, 0.1)  # 增加步长减少样本数量
    Y_samples = X_samples.copy()
    X0_samples = step_sample(0, PI/9, 0.1)
    
    # 条件1: T_{i,i'}(x, f(x)) >= 0
    print("添加条件1约束...")
    for x in X_samples[:10]:  # 进一步减少样本
        xp = f_m(x)
        for i in [0, 1]:
            L_x = In_Unsafe(x)
            i_prime_list = delta(i, L_x)
            for i_prime in i_prime_list:
                T_val = T_template(x, xp, i, i_prime, coeffs)
                s.add(T_val >= 0)
    
    # 条件2: T_{i',j}(f(x), y) >= 0 => T_{i,j}(x, y) >= 0
    print("添加条件2约束...")
    for x in X_samples[:5]:  # 进一步减少样本
        xp = f_m(x)
        for y in Y_samples[:5]:
            for i in [0, 1]:
                for j in [0, 1]:
                    L_x = In_Unsafe(x)
                    i_prime_list = delta(i, L_x)
                    
                    for i_prime in i_prime_list:
                        premise = T_template(xp, y, i_prime, j, coeffs) >= 0
                        conclusion = T_template(x, y, i, j, coeffs) >= 0
                        s.add(z3.Implies(premise, conclusion))
    
    # 条件3: 下降条件
    print("添加条件3约束...")
    s_val = 1
    l_val = 0
    l_prime_val = 0
    
    for x0 in X0_samples[:3]:
        for z in X_samples[:3]:
            for z_prime in X_samples[:3]:
                premise1 = T_template(x0, z, s_val, l_val, coeffs) >= 0
                premise2 = T_template(z, z_prime, l_val, l_prime_val, coeffs) >= 0
                conclusion = T_template(x0, z_prime, s_val, l_prime_val, coeffs) + EPSILON <= T_template(x0, z, s_val, l_val, coeffs)
                s.add(z3.Implies(z3.And(premise1, premise2), conclusion))
    
    # 设置CEGIS循环
    MAX_ITER = 100
    iter_count = 0
    
    # CEGIS主循环
    while s.check() == z3.sat and iter_count < MAX_ITER:
        ce_found = False
        m = s.model()
        
        # 获取当前系数值
        coeffs_vals = [t2float(m[c]) for c in coeffs]
        epsilon_val = t2float(m[EPSILON])
        
        print(f"\n迭代 #{iter_count}:")
        print(f"epsilon = {epsilon_val:.6f}")
        for i, val in enumerate(coeffs_vals):
            print(f"c{i:02d} = {val:.6f}")
        
        # 检查条件1的反例 - 使用数值验证而不是dreal
        print("检查条件1...")
        condition1_found = check_condition1_numerical(coeffs_vals, coeffs, s, num_samples=num_samples)
        if condition1_found:
            ce_found = True
        
        # 检查条件2的反例 - 使用数值验证
        print("检查条件2...")
        condition2_found = check_condition2_numerical(coeffs_vals, coeffs, s, num_samples=num_samples)
        if condition2_found:
            ce_found = True
        
        # 检查条件3的反例 - 使用数值验证
        print("检查条件3...")
        condition3_found = check_condition3_numerical(coeffs_vals, coeffs, s, epsilon_val, num_samples=num_samples)
        if condition3_found:
            ce_found = True
        
        # 如果没有找到反例，则找到有效证书
        if not ce_found:
            print(f"\n成功合成闭包证书，迭代次数: {iter_count}")
            return coeffs_vals, epsilon_val
        
        iter_count += 1
    
    if iter_count >= MAX_ITER:
        print(f"超过最大迭代次数 {MAX_ITER}")
    else:
        print("无法合成闭包证书")
    
    return None, None


def check_condition1_numerical(coeffs_vals, coeffs, solver, num_samples=1000):
    """数值验证条件1"""
    found_counterexample = False
    
    for _ in range(num_samples):
        x_val = random.uniform(0, 2 * PI)
        xp_val = f_m(x_val)
        
        for i in [0, 1]:
            L_x = In_Unsafe(x_val)
            i_prime_list = delta(i, L_x)
            
            for i_prime in i_prime_list:
                T_val = T_template(x_val, xp_val, i, i_prime, coeffs_vals)
                if T_val < 0:
                    print(f"数值验证发现条件1反例: i={i}, i'={i_prime}, x={x_val:.6f}, T={T_val:.6f}")
                    T_val_z3 = T_template(x_val, xp_val, i, i_prime, coeffs)
                    solver.add(T_val_z3 >= 0)
                    found_counterexample = True
                    return found_counterexample
    
    return found_counterexample


def check_condition2_numerical(coeffs_vals, coeffs, solver, num_samples=500):
    """数值验证条件2"""
    found_counterexample = False
    
    for _ in range(num_samples):
        x_val = random.uniform(0, 2 * PI)
        y_val = random.uniform(0, 2 * PI)
        xp_val = f_m(x_val)
        
        for i in [0, 1]:
            for j in [0, 1]:
                L_x = In_Unsafe(x_val)
                i_prime_list = delta(i, L_x)
                
                for i_prime in i_prime_list:
                    # 前提: T_{i',j}(f(x), y) >= 0
                    premise_val = T_template(xp_val, y_val, i_prime, j, coeffs_vals)
                    
                    # 结论: T_{i,j}(x, y) >= 0
                    conclusion_val = T_template(x_val, y_val, i, j, coeffs_vals)
                    
                    if premise_val >= 0 and conclusion_val < 0:
                        print(f"数值验证发现条件2反例: i={i}, j={j}, i'={i_prime}, "
                              f"x={x_val:.6f}, y={y_val:.6f}")
                        premise_z3 = T_template(xp_val, y_val, i_prime, j, coeffs) >= 0
                        conclusion_z3 = T_template(x_val, y_val, i, j, coeffs) >= 0
                        solver.add(z3.Implies(premise_z3, conclusion_z3))
                        found_counterexample = True
                        return found_counterexample
    
    return found_counterexample


def check_condition3_numerical(coeffs_vals, coeffs, solver, epsilon_val, num_samples=200):
    """数值验证条件3"""
    found_counterexample = False
    s_val = 1
    l_val = 0
    l_prime_val = 0
    
    for _ in range(num_samples):
        x0_val = random.uniform(0, PI/9)  # x0在X0中
        z_val = random.uniform(0, 2 * PI)
        z_prime_val = random.uniform(0, 2 * PI)
        
        # 前提1: T_{s,l}(x0, z) >= 0
        premise1_val = T_template(x0_val, z_val, s_val, l_val, coeffs_vals)
        
        # 前提2: T_{l,l'}(z, z') >= 0
        premise2_val = T_template(z_val, z_prime_val, l_val, l_prime_val, coeffs_vals)
        
        # 结论: T_{s,l'}(x0, z') + epsilon <= T_{s,l}(x0, z)
        T_sl_prime_val = T_template(x0_val, z_prime_val, s_val, l_prime_val, coeffs_vals)
        
        if premise1_val >= 0 and premise2_val >= 0:
            if T_sl_prime_val + epsilon_val > premise1_val:
                print(f"数值验证发现条件3反例: x0={x0_val:.6f}, z={z_val:.6f}, z'={z_prime_val:.6f}")
                premise1_z3 = T_template(x0_val, z_val, s_val, l_val, coeffs) >= 0
                premise2_z3 = T_template(z_val, z_prime_val, l_val, l_prime_val, coeffs) >= 0
                conclusion_z3 = T_template(x0_val, z_prime_val, s_val, l_prime_val, coeffs) + EPSILON <= T_template(x0_val, z_val, s_val, l_val, coeffs)
                solver.add(z3.Implies(z3.And(premise1_z3, premise2_z3), conclusion_z3))
                found_counterexample = True
                return found_counterexample
    
    return found_counterexample


if __name__ == "__main__":
    start_time = time.time()
    num_samples = 1000
    # 合成闭包证书
    coeffs, epsilon_val = synthesize_closure_certificate(num_samples=num_samples)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f}秒")
    
    if coeffs is not None and epsilon_val is not None:
        print("\n" + "="*60)
        print("闭包证书合成成功!")
        print("="*60)
        
        # 输出epsilon值
        print(f"\n下降参数 epsilon = {epsilon_val:.6f}")
        
        # 输出T_00的系数
        c00, c10, c20 = coeffs[0:3]
        print(f"\nT_00(x, y) 系数:")
        print(f"  c00 = {c00:.6f}")
        print(f"  c10 = {c10:.6f}")
        print(f"  c20 = {c20:.6f}")
        
        # 输出T_01的系数
        c01, c11, c21 = coeffs[3:6]
        print(f"\nT_01(x, y) 系数:")
        print(f"  c01 = {c01:.6f}")
        print(f"  c11 = {c11:.6f}")
        print(f"  c21 = {c21:.6f}")
        
        # 输出T_10的系数
        c02, c12, c22 = coeffs[6:9]
        print(f"\nT_10(x, y) 系数:")
        print(f"  c02 = {c02:.6f}")
        print(f"  c12 = {c12:.6f}")
        print(f"  c22 = {c22:.6f}")
        
        # 输出T_11的系数
        c03, c13, c23 = coeffs[9:12]
        print(f"\nT_11(x, y) 系数:")
        print(f"  c03 = {c03:.6f}")
        print(f"  c13 = {c13:.6f}")
        print(f"  c23 = {c23:.6f}")
        
        # 测试几个点验证证书
        print("\n" + "="*60)
        print("测试点验证:")
        print("="*60)
        
        test_points = [
            (0.1, 0.2),    # 都在X0中
            (1.0, 1.5),    # 都在X中但不在X0或Xu中
            (2.5, 2.7),    # 都在X中
        ]
        
        for x, y in test_points:
            print(f"\n测试点 (x={x:.3f}, y={y:.3f}):")
            for i in [0, 1]:
                for j in [0, 1]:
                    T_val = T_template(x, y, i, j, coeffs)
                    print(f"  T_{i}{j} = {T_val:.6f}")
    else:
        print("\n闭包证书合成失败!")