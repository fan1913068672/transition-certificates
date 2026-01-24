import math
import dreal
import numpy as np
import time
from dreal import Variable, And, Not, if_then_else

# ==================== 系统参数定义 ====================
TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926

# ==================== 数值方法合成的证书参数 ====================
# 从您的合成结果获取的参数
EPSILON = 0.000510

# 系数数组 (20个系数，每个T_ij有5个系数)
coeffs = [
    # T_00: c00, c10_x0, c20_y0, c10_xu, c20_yu
    0.003058, 0.000510, -0.000510, 0.000510, -0.000510,
    
    # T_01: c01, c11_x0, c21_y0, c11_xu, c21_yu
    -0.000510, -0.000510, -1.227905, -0.000510, -1.227905,
    
    # T_10: c02, c12_x0, c22_y0, c12_xu, c22_yu
    -0.023626, 0.010745, -0.000510, 0.010745, -0.000510,
    
    # T_11: c03, c13_x0, c23_y0, c13_xu, c23_yu
    100.000000, -0.000510, -46.250723, -0.000510, -46.250723
]

# 检查系数数量
assert len(coeffs) == 20, f"期望20个系数，但得到{len(coeffs)}个"

# ==================== 集合定义 ====================
def In_X_Cond(x):
    """定义状态空间 X 的条件: x ∈ [0, 2π]"""
    return And(x >= 0, x <= PI * 2)

def In_X0_Cond(x):
    """定义初始状态集 X0 的条件: x ∈ [0, π/9]"""
    return And(x >= 0, x <= PI / 9)

def In_Unsafe_Cond(x):
    """定义不安全状态集 Xu 的条件: x ∈ [7π/9, 8π/9]"""
    return And(x >= 7 * PI / 9, x <= 8 * PI / 9)

# ==================== 系统动力学 ====================
def f_t(x):
    """连续时间动力学函数 (dreal版本)"""
    return x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x ** 2 + 1.69

def f_m(x):
    """离散时间动力学函数 (数值版本)"""
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69

# ==================== 闭包证书类 ====================
class ClosureCertificateVerifier:
    def __init__(self, coeffs, epsilon, tolerance=1e-10):
        """初始化证书验证器"""
        self.coeffs = coeffs
        self.epsilon = epsilon
        self.tol = tolerance
        self.ctx = dreal.Context()
        self.ctx.config.precision = 1e-6  # 设置求解精度
        
    def _region_type(self, x):
        """返回x在X0、Xu、X_other中的条件"""
        in_x0 = In_X0_Cond(x)
        in_xu = In_Unsafe_Cond(x)
        in_x_other = And(Not(in_x0), Not(in_xu))
        return in_x0, in_xu, in_x_other
    
    def T_dreal(self, x, y, i, j):
        """dreal版本的T函数，处理区域分支"""
        # 获取x和y的区域条件
        x_in_x0, x_in_xu, x_in_other = self._region_type(x)
        y_in_x0, y_in_xu, y_in_other = self._region_type(y)
        
        # 根据i,j确定基础索引
        if i == 0 and j == 0:
            base_idx = 0
        elif i == 0 and j == 1:
            base_idx = 5
        elif i == 1 and j == 0:
            base_idx = 10
        elif i == 1 and j == 1:
            base_idx = 15
        else:
            raise ValueError(f"无效状态对: ({i}, {j})")
        
        # 获取系数
        c0 = self.coeffs[base_idx]
        cx0 = self.coeffs[base_idx + 1]  # x∈X0时的系数
        cy0 = self.coeffs[base_idx + 2]  # y∈X0时的系数
        cxu = self.coeffs[base_idx + 3]  # x∈Xu时的系数
        cyu = self.coeffs[base_idx + 4]  # y∈Xu时的系数
        
        # 构建基于区域的分支表达式
        # 首先处理x的分支
        result = if_then_else(
            x_in_x0,
            # x ∈ X0
            if_then_else(
                y_in_x0,
                c0 + cx0 * x + cy0 * y,         # y ∈ X0
                if_then_else(
                    y_in_xu,
                    c0 + cx0 * x + cyu * y,     # y ∈ Xu
                    c0 + cx0 * x                # y ∈ X_other
                )
            ),
            if_then_else(
                x_in_xu,
                # x ∈ Xu
                if_then_else(
                    y_in_x0,
                    c0 + cxu * x + cy0 * y,     # y ∈ X0
                    if_then_else(
                        y_in_xu,
                        c0 + cxu * x + cyu * y, # y ∈ Xu
                        c0 + cxu * x            # y ∈ X_other
                    )
                ),
                # x ∈ X_other
                if_then_else(
                    y_in_x0,
                    c0 + cy0 * y,               # y ∈ X0
                    if_then_else(
                        y_in_xu,
                        c0 + cyu * y,           # y ∈ Xu
                        c0                      # y ∈ X_other
                    )
                )
            )
        )
        
        return result
    
    def T_numeric(self, x_val, y_val, i, j):
        """数值版本的T函数，用于验证"""
        # 确定区域类型
        def region_type_val(val):
            if 0 <= val <= PI/9:
                return 0  # X0
            elif 7*PI/9 <= val <= 8*PI/9:
                return 1  # Xu
            else:
                return 2  # X_other
        
        x_region = region_type_val(x_val)
        y_region = region_type_val(y_val)
        
        # 根据i,j确定基础索引
        if i == 0 and j == 0:
            base_idx = 0
        elif i == 0 and j == 1:
            base_idx = 5
        elif i == 1 and j == 0:
            base_idx = 10
        elif i == 1 and j == 1:
            base_idx = 15
        else:
            raise ValueError(f"无效状态对: ({i}, {j})")
        
        # 获取系数
        c0 = self.coeffs[base_idx]
        cx0 = self.coeffs[base_idx + 1]
        cy0 = self.coeffs[base_idx + 2]
        cxu = self.coeffs[base_idx + 3]
        cyu = self.coeffs[base_idx + 4]
        
        # 根据区域计算T值
        if x_region == 0:  # x ∈ X0
            if y_region == 0:  # y ∈ X0
                return c0 + cx0 * x_val + cy0 * y_val
            elif y_region == 1:  # y ∈ Xu
                return c0 + cx0 * x_val + cyu * y_val
            else:  # y ∈ X_other
                return c0 + cx0 * x_val
        elif x_region == 1:  # x ∈ Xu
            if y_region == 0:  # y ∈ X0
                return c0 + cxu * x_val + cy0 * y_val
            elif y_region == 1:  # y ∈ Xu
                return c0 + cxu * x_val + cyu * y_val
            else:  # y ∈ X_other
                return c0 + cxu * x_val
        else:  # x ∈ X_other
            if y_region == 0:  # y ∈ X0
                return c0 + cy0 * y_val
            elif y_region == 1:  # y ∈ Xu
                return c0 + cyu * y_val
            else:  # y ∈ X_other
                return c0

# ==================== 自动机定义 ====================
class Automaton:
    @staticmethod
    def delta(i, L_x):
        """模态转移函数"""
        if i == 1:
            return [0] if L_x else [1]
        elif i == 0:
            return [0]
        else:
            raise ValueError(f"无效状态: {i}")
    
    @staticmethod
    def L(x):
        """标签函数: 返回x是否在不安全集中"""
        return In_Unsafe_Cond(x)
    
    @staticmethod
    def L_numeric(x_val):
        """数值版本的标签函数"""
        return 7*PI/9 <= x_val <= 8*PI/9

# ==================== 验证函数 ====================
def verify_condition1(verifier, verbose=False):
    """验证条件1: T_{i,i'}(x, f(x)) >= 0"""
    print("\n" + "="*60)
    print("验证条件1: T_{i,i'}(x, f(x)) >= 0")
    print("="*60)
    
    ctx = dreal.Context()
    ctx.config.precision = 1e-6
    
    all_violations = []
    total_checks = 0
    
    # 对所有可能的转移进行验证
    for i in [0, 1]:
        for L_x_bool in [True, False]:  # L(x)为真或假
            i_prime_list = Automaton.delta(i, L_x_bool)
            
            for i_prime in i_prime_list:
                total_checks += 1
                
                # 创建变量
                x = Variable("x")
                ctx.DeclareVariable(x, 0, 2*PI)
                
                # 添加约束
                ctx.Assert(In_X_Cond(x))
                
                # 根据L_x添加约束
                if L_x_bool:
                    ctx.Assert(In_Unsafe_Cond(x))
                else:
                    ctx.Assert(Not(In_Unsafe_Cond(x)))
                
                # 检查T < 0的情况
                T_val = verifier.T_dreal(x, f_t(x), i, i_prime)
                ctx.Assert(T_val < -verifier.tol)
                
                # 求解
                result = ctx.CheckSat()
                
                if result is not None:
                    # 提取反例
                    x_val = result[x].mid()
                    f_x_val = f_m(x_val)
                    
                    # 计算实际的T值
                    T_actual = verifier.T_numeric(x_val, f_x_val, i, i_prime)
                    
                    violation_info = {
                        'condition': 'condition1',
                        'i': i,
                        'i_prime': i_prime,
                        'L_x': L_x_bool,
                        'x': x_val,
                        'f_x': f_x_val,
                        'T_value': T_actual
                    }
                    all_violations.append(violation_info)
                    
                    if verbose:
                        region = "Xu" if L_x_bool else "not Xu"
                        print(f"  ❌ 反例找到: i={i}, i'={i_prime}, x∈{region}")
                        print(f"     x = {x_val:.6f}")
                        print(f"     f(x) = {f_x_val:.6f}")
                        print(f"     T_{i}{i_prime}(x, f(x)) = {T_actual:.6f}")
                        print()
                
                # 重置上下文
                ctx = dreal.Context()
                ctx.config.precision = 1e-6
    
    # 输出结果
    if not all_violations:
        print("  ✅ 条件1通过验证")
    else:
        print(f"  ❌ 发现 {len(all_violations)}/{total_checks} 个反例")
        
        # 打印前5个反例
        for i, vi in enumerate(all_violations[:5]):
            region = "Xu" if vi['L_x'] else "not Xu"
            print(f"    反例 {i+1}: i={vi['i']}, i'={vi['i_prime']}, x∈{region}")
            print(f"      x={vi['x']:.6f}, f(x)={vi['f_x']:.6f}")
            print(f"      T={vi['T_value']:.6f}")
            
        if len(all_violations) > 5:
            print(f"    ... 还有 {len(all_violations)-5} 个反例未显示")
    
    return all_violations

def verify_condition2(verifier, verbose=False):
    """验证条件2: T_{i',j}(f(x), y) >= 0 ⇒ T_{i,j}(x, y) >= 0"""
    print("\n" + "="*60)
    print("验证条件2: T_{i',j}(f(x), y) >= 0 ⇒ T_{i,j}(x, y) >= 0")
    print("="*60)
    
    all_violations = []
    total_checks = 0
    
    # 对所有可能的组合进行验证
    for i in [0, 1]:
        for j in [0, 1]:
            for L_x_bool in [True, False]:
                i_prime_list = Automaton.delta(i, L_x_bool)
                
                for i_prime in i_prime_list:
                    total_checks += 1
                    
                    ctx = dreal.Context()
                    ctx.config.precision = 1e-6
                    
                    # 创建变量
                    x = Variable("x")
                    y = Variable("y")
                    ctx.DeclareVariable(x, 0, 2*PI)
                    ctx.DeclareVariable(y, 0, 2*PI)
                    
                    # 添加约束
                    ctx.Assert(In_X_Cond(x))
                    ctx.Assert(In_X_Cond(y))
                    
                    # 根据L_x添加约束
                    if L_x_bool:
                        ctx.Assert(In_Unsafe_Cond(x))
                    else:
                        ctx.Assert(Not(In_Unsafe_Cond(x)))
                    
                    # 构建条件: 前提成立但结论不成立
                    premise = verifier.T_dreal(f_t(x), y, i_prime, j) >= 0
                    conclusion = verifier.T_dreal(x, y, i, j) < -verifier.tol
                    
                    ctx.Assert(And(premise, conclusion))
                    
                    # 求解
                    result = ctx.CheckSat()
                    
                    if result is not None:
                        # 提取反例
                        x_val = result[x].mid()
                        y_val = result[y].mid()
                        f_x_val = f_m(x_val)
                        
                        # 计算实际的T值
                        T_premise = verifier.T_numeric(f_x_val, y_val, i_prime, j)
                        T_conclusion = verifier.T_numeric(x_val, y_val, i, j)
                        
                        violation_info = {
                            'condition': 'condition2',
                            'i': i,
                            'j': j,
                            'i_prime': i_prime,
                            'L_x': L_x_bool,
                            'x': x_val,
                            'y': y_val,
                            'f_x': f_x_val,
                            'T_premise': T_premise,
                            'T_conclusion': T_conclusion
                        }
                        all_violations.append(violation_info)
                        
                        if verbose:
                            region = "Xu" if L_x_bool else "not Xu"
                            print(f"  ❌ 反例找到: i={i}, j={j}, i'={i_prime}, x∈{region}")
                            print(f"     x = {x_val:.6f}, f(x) = {f_x_val:.6f}, y = {y_val:.6f}")
                            print(f"     T_{i_prime}{j}(f(x), y) = {T_premise:.6f}")
                            print(f"     T_{i}{j}(x, y) = {T_conclusion:.6f}")
                            print()
    
    # 输出结果
    if not all_violations:
        print("  ✅ 条件2通过验证")
    else:
        print(f"  ❌ 发现 {len(all_violations)}/{total_checks} 个反例")
        
        # 打印前3个反例
        for i, vi in enumerate(all_violations[:3]):
            region = "Xu" if vi['L_x'] else "not Xu"
            print(f"    反例 {i+1}: i={vi['i']}, j={vi['j']}, i'={vi['i_prime']}, x∈{region}")
            print(f"      x={vi['x']:.6f}, f(x)={vi['f_x']:.6f}, y={vi['y']:.6f}")
            print(f"      T_{vi['i_prime']}{vi['j']}(f(x),y)={vi['T_premise']:.6f}")
            print(f"      T_{vi['i']}{vi['j']}(x,y)={vi['T_conclusion']:.6f}")
            
        if len(all_violations) > 3:
            print(f"    ... 还有 {len(all_violations)-3} 个反例未显示")
    
    return all_violations

def verify_condition3(verifier, verbose=False):
    """验证条件3: 下降条件"""
    print("\n" + "="*60)
    print("验证条件3: 下降条件")
    print("="*60)
    
    ctx = dreal.Context()
    ctx.config.precision = 1e-6
    
    # 定义变量
    x0 = Variable("x0")
    z = Variable("z")
    z_prime = Variable("z_prime")
    
    ctx.DeclareVariable(x0, 0, PI/9)      # x0 ∈ X0
    ctx.DeclareVariable(z, 0, 2*PI)      # z ∈ X
    ctx.DeclareVariable(z_prime, 0, 2*PI)  # z' ∈ X
    
    # 定义自动机状态
    s = 1  # 初始状态
    l = 0  # 接受状态
    l_prime = 0  # 接受状态
    
    # 构建条件3的前提和结论
    premise1 = verifier.T_dreal(x0, z, s, l) >= 0
    premise2 = verifier.T_dreal(z, z_prime, l, l_prime) >= 0
    
    # 结论的否定: T_{s,l'}(x0, z') + ε > T_{s,l}(x0, z)
    T_sl_prime = verifier.T_dreal(x0, z_prime, s, l_prime)
    T_sl = verifier.T_dreal(x0, z, s, l)
    conclusion_neg = T_sl_prime + verifier.epsilon - verifier.tol > T_sl
    
    # 检查是否存在反例
    ctx.Assert(And(premise1, premise2, conclusion_neg))
    
    result = ctx.CheckSat()
    
    if result is not None:
        # 提取反例
        x0_val = result[x0].mid()
        z_val = result[z].mid()
        z_prime_val = result[z_prime].mid()
        
        # 计算实际的T值
        T_sl_val = verifier.T_numeric(x0_val, z_val, s, l)
        T_ll_val = verifier.T_numeric(z_val, z_prime_val, l, l_prime)
        T_sl_prime_val = verifier.T_numeric(x0_val, z_prime_val, s, l_prime)
        
        violation_info = {
            'condition': 'condition3',
            's': s,
            'l': l,
            'l_prime': l_prime,
            'x0': x0_val,
            'z': z_val,
            'z_prime': z_prime_val,
            'T_sl': T_sl_val,
            'T_ll': T_ll_val,
            'T_sl_prime': T_sl_prime_val
        }
        
        if verbose:
            print(f"  ❌ 反例找到:")
            print(f"     x0 = {x0_val:.6f}")
            print(f"     z = {z_val:.6f}")
            print(f"     z' = {z_prime_val:.6f}")
            print(f"     T_{s}{l}(x0, z) = {T_sl_val:.6f}")
            print(f"     T_{l}{l_prime}(z, z') = {T_ll_val:.6f}")
            print(f"     T_{s}{l_prime}(x0, z') = {T_sl_prime_val:.6f}")
            print(f"     ε = {verifier.epsilon:.6f}")
            print(f"     T_sl' + ε = {T_sl_prime_val + verifier.epsilon:.6f}")
            print(f"     T_sl = {T_sl_val:.6f}")
        
        return [violation_info]
    else:
        print("  ✅ 条件3通过验证")
        return []

# ==================== 随机测试 ====================
def random_test(verifier, num_tests=10000, verbose=False):
    """随机测试证书的有效性"""
    print("\n" + "="*60)
    print(f"随机测试 ({num_tests} 个随机点)")
    print("="*60)
    
    violations = []
    
    for test_idx in range(num_tests):
        # 生成随机点
        x = np.random.uniform(0, 2*PI)
        y = np.random.uniform(0, 2*PI)
        x0 = np.random.uniform(0, PI/9)
        z = np.random.uniform(0, 2*PI)
        z_prime = np.random.uniform(0, 2*PI)
        
        # 测试条件1的所有可能情况
        for i in [0, 1]:
            L_x = Automaton.L_numeric(x)
            i_prime_list = Automaton.delta(i, L_x)
            
            for i_prime in i_prime_list:
                T_val = verifier.T_numeric(x, f_m(x), i, i_prime)
                if T_val < -verifier.tol:
                    violations.append({
                        'test': 'condition1',
                        'x': x, 'f_x': f_m(x),
                        'i': i, 'i_prime': i_prime,
                        'T_val': T_val
                    })
        
        # 测试条件2
        for i in [0, 1]:
            for j in [0, 1]:
                L_x = Automaton.L_numeric(x)
                i_prime_list = Automaton.delta(i, L_x)
                
                for i_prime in i_prime_list:
                    premise = verifier.T_numeric(f_m(x), y, i_prime, j)
                    conclusion = verifier.T_numeric(x, y, i, j)
                    
                    if premise >= 0 and conclusion < -verifier.tol:
                        violations.append({
                            'test': 'condition2',
                            'x': x, 'y': y, 'f_x': f_m(x),
                            'i': i, 'j': j, 'i_prime': i_prime,
                            'premise': premise, 'conclusion': conclusion
                        })
        
        # 测试条件3
        s, l, l_prime = 1, 0, 0
        premise1 = verifier.T_numeric(x0, z, s, l)
        premise2 = verifier.T_numeric(z, z_prime, l, l_prime)
        
        if premise1 >= 0 and premise2 >= 0:
            left = verifier.T_numeric(x0, z_prime, s, l_prime) + EPSILON
            right = premise1
            
            if left > right + verifier.tol:
                violations.append({
                    'test': 'condition3',
                    'x0': x0, 'z': z, 'z_prime': z_prime,
                    'premise1': premise1, 'premise2': premise2,
                    'left': left, 'right': right
                })
        
        # 显示进度
        if verbose and (test_idx + 1) % 1000 == 0:
            print(f"  已完成 {test_idx + 1} 个测试点")
    
    if not violations:
        print(f"  ✅ 随机测试通过 ({num_tests} 个点)")
    else:
        print(f"  ⚠️  发现 {len(violations)} 个潜在问题")
        for i, vi in enumerate(violations[:5]):
            print(f"    问题 {i+1}: {vi['test']}")
    
    return violations

# ==================== 主验证函数 ====================
def verify_certificate():
    """主验证函数"""
    print("基于dReal验证数值方法合成的闭包证书")
    print("="*60)
    print(f"ε = {EPSILON:.6f}")
    print("="*60)
    
    start_time = time.time()
    
    # 创建验证器
    verifier = ClosureCertificateVerifier(coeffs, EPSILON, tolerance=0)
    
    # 执行验证
    violations_cond1 = verify_condition1(verifier, verbose=True)
    violations_cond2 = verify_condition2(verifier, verbose=True)
    violations_cond3 = verify_condition3(verifier, verbose=True)
    
    # 随机测试
    violations_random = random_test(verifier, num_tests=10000, verbose=True)
    
    end_time = time.time()
    
    # 汇总结果
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    
    all_violations = (
        len(violations_cond1) + 
        len(violations_cond2) + 
        len(violations_cond3) + 
        len(violations_random)
    )
    
    if all_violations == 0:
        print("🎉 证书验证成功！所有条件都满足。")
        print(f"  - 条件1: 通过")
        print(f"  - 条件2: 通过")
        print(f"  - 条件3: 通过")
        print(f"  - 随机测试: 通过")
    else:
        print("⚠️ 证书验证未通过，发现以下问题:")
        if violations_cond1:
            print(f"  - 条件1: 发现 {len(violations_cond1)} 个反例")
        else:
            print(f"  - 条件1: 通过")
            
        if violations_cond2:
            print(f"  - 条件2: 发现 {len(violations_cond2)} 个反例")
        else:
            print(f"  - 条件2: 通过")
            
        if violations_cond3:
            print(f"  - 条件3: 发现 {len(violations_cond3)} 个反例")
        else:
            print(f"  - 条件3: 通过")
            
        if violations_random:
            print(f"  - 随机测试: 发现 {len(violations_random)} 个问题")
        else:
            print(f"  - 随机测试: 通过")
    
    print(f"\n验证耗时: {end_time - start_time:.2f} 秒")
    
    return all_violations == 0

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 验证证书
    is_valid = verify_certificate()
    
    # 保存结果
    with open("certificate_verification_result.txt", "w") as f:
        f.write("闭包证书验证结果\n")
        f.write("="*60 + "\n")
        f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ε = {EPSILON:.6f}\n")
        f.write("="*60 + "\n\n")
        
        f.write("系数值:\n")
        for i, c in enumerate(coeffs):
            f.write(f"  c{i:02d} = {c:.6f}\n")
        
        f.write("\n验证结果: ")
        if is_valid:
            f.write("✅ 证书有效\n")
        else:
            f.write("⚠️ 证书存在反例\n")
    
    print(f"\n详细结果已保存到 'certificate_verification_result.txt'")