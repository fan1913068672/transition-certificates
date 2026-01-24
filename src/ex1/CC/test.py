'''
闭包证书验证测试 - 修正版本
准确验证闭包证书的三个条件
'''

import math
import numpy as np
import random

# ==================== 系统参数定义 ====================
TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926
TOL = 1e-6  # 数值容差

# ==================== 集合定义 ====================
def In_X0(x):
    """检查数值x是否在初始集中"""
    return 0 <= x <= PI / 9

def In_Unsafe(x):
    """检查数值x是否在不安全集中"""
    return 7 * PI / 9 <= x <= 8 * PI / 9

# ==================== 系统动力学 ====================
def f_m(x):
    """离散时间动力学函数"""
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69

# ==================== 辅助函数 ====================
def region_type(x):
    """确定x所在的区域类型"""
    if In_X0(x):
        return 0
    elif In_Unsafe(x):
        return 1
    else:
        return 2

# ==================== 闭包证书类 ====================
class ClosureCertificate:
    def __init__(self, coeffs):
        self.coeffs = coeffs
    
    def T(self, x, y, i, j):
        """计算T_ij(x, y)"""
        x_region = region_type(x)
        y_region = region_type(y)
        
        # 根据i,j确定系数索引
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
        
        c0, cx0, cy0, cxu, cyu = self.coeffs[base_idx:base_idx+5]
        
        # 根据区域选择表达式
        if x_region == 0:  # x ∈ X0
            if y_region == 0:  # y ∈ X0
                return c0 + cx0 * x + cy0 * y
            elif y_region == 1:  # y ∈ Xu
                return c0 + cxu * x + cyu * y
            else:  # y ∈ X_other
                return c0
        elif x_region == 1:  # x ∈ Xu
            if y_region == 0:  # y ∈ X0
                return c0 + cx0 * x + cy0 * y
            elif y_region == 1:  # y ∈ Xu
                return c0 + cxu * x + cyu * y
            else:  # y ∈ X_other
                return c0 + cxu * x
        else:  # x ∈ X_other
            if y_region == 0:  # y ∈ X0
                return c0 + cx0 * x
            elif y_region == 1:  # y ∈ Xu
                return c0 + cyu * y
            else:  # y ∈ X_other
                return c0

# ==================== 自动机类 ====================
class Automaton:
    """NBA自动机"""
    
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
    def get_label(x):
        """标签函数 L: X → 2^{AP}，这里简化为是否在Xu中"""
        return In_Unsafe(x)
    
    @staticmethod
    def get_initial_state():
        """初始状态 q0 = 1"""
        return 1
    
    @staticmethod
    def is_accepting_state(q):
        """接受状态 Acc = {0}"""
        return q == 0

# ==================== 测试类 ====================
class ClosureCertificateTester:
    def __init__(self, coeffs, epsilon, num_tests=10000):
        self.coeffs = coeffs  # 存储系数
        self.cert = ClosureCertificate(coeffs)
        self.epsilon = epsilon
        self.num_tests = num_tests
        self.tol = TOL
        
    def test_condition1(self):
        """
        测试条件1: T_{i,i'}(x, f(x)) >= 0
        其中 i' ∈ δ(i, L(x)), x ∈ X, i ∈ Q
        """
        print("="*60)
        print("测试条件1: 相邻转移证书值非负")
        print("="*60)
        
        violations = []
        
        for _ in range(self.num_tests):
            x = random.uniform(0, 2*PI)
            xp = f_m(x)
            L_x = Automaton.get_label(x)
            
            for i in [0, 1]:
                i_prime_list = Automaton.delta(i, L_x)
                
                for i_prime in i_prime_list:
                    T_val = self.cert.T(x, xp, i, i_prime)
                    
                    if T_val < -self.tol:
                        violations.append({
                            'x': x, 'xp': xp, 'i': i, 'i_prime': i_prime,
                            'T_val': T_val, 'L_x': L_x
                        })
        
        print(f"测试次数: {self.num_tests * 2 * 2}")  # 2个x值 * 2个i * 2个可能的i'
        print(f"违反次数: {len(violations)}")
        
        if violations:
            print("\n违反示例:")
            for i, v in enumerate(violations[:5]):  # 只显示前5个
                print(f"  示例{i+1}: x={v['x']:.4f}, f(x)={v['xp']:.4f}, "
                      f"i={v['i']}, i'={v['i_prime']}, L(x)={v['L_x']}")
                print(f"    T_{v['i']}{v['i_prime']}(x, f(x)) = {v['T_val']:.6f}")
        
        return len(violations) == 0
    
    def test_condition2(self):
        """
        测试条件2: T_{i',j}(f(x), y) >= 0 ⇒ T_{i,j}(x, y) >= 0
        其中 i' ∈ δ(i, L(x)), x, y ∈ X, i, j ∈ Q
        
        这表示：如果从(f(x), i')能到达(y, j)，那么从(x, i)也能到达(y, j)
        """
        print("\n" + "="*60)
        print("测试条件2: 迁移蕴含关系")
        print("="*60)
        
        violations = []
        
        for _ in range(self.num_tests // 10):  # 减少测试数量
            x = random.uniform(0, 2*PI)
            y = random.uniform(0, 2*PI)
            xp = f_m(x)
            L_x = Automaton.get_label(x)
            
            for i in [0, 1]:
                i_prime_list = Automaton.delta(i, L_x)
                
                for j in [0, 1]:
                    for i_prime in i_prime_list:
                        T_i_prime_j = self.cert.T(xp, y, i_prime, j)
                        T_i_j = self.cert.T(x, y, i, j)
                        
                        # 检查蕴含关系
                        if T_i_prime_j >= -self.tol and T_i_j < -self.tol:
                            violations.append({
                                'x': x, 'y': y, 'xp': xp,
                                'i': i, 'j': j, 'i_prime': i_prime,
                                'T_i_prime_j': T_i_prime_j,
                                'T_i_j': T_i_j,
                                'L_x': L_x
                            })
        
        print(f"测试次数: {(self.num_tests // 10) * 2 * 2 * 2}")  # 减少的测试
        print(f"违反次数: {len(violations)}")
        
        if violations:
            print("\n违反示例:")
            for i, v in enumerate(violations[:5]):
                print(f"  示例{i+1}: x={v['x']:.4f}, y={v['y']:.4f}, "
                      f"f(x)={v['xp']:.4f}, i={v['i']}, j={v['j']}, i'={v['i_prime']}")
                print(f"    T_{v['i_prime']}{v['j']}(f(x), y) = {v['T_i_prime_j']:.6f} >= 0")
                print(f"    T_{v['i']}{v['j']}(x, y) = {v['T_i_j']:.6f} < 0")
        
        return len(violations) == 0
    
    def test_condition3(self):
        """
        测试条件3: 下降条件
        (T_{s,l}(x0, z) >= 0 and T_{l,l'}(z, z') >= 0) 
        ⇒ (T_{s,l'}(x0, z') + ε <= T_{s,l}(x0, z))
        
        其中 s = q0 = 1, l, l' ∈ Acc = {0}, x0 ∈ X0, z, z' ∈ X
        """
        print("\n" + "="*60)
        print("测试条件3: 下降条件")
        print("="*60)
        
        s = Automaton.get_initial_state()  # s = 1
        l = 0  # 接受状态
        l_prime = 0  # 接受状态
        
        violations = []
        
        for _ in range(self.num_tests // 20):  # 进一步减少测试数量
            # 从X0中采样x0
            x0 = random.uniform(0, PI/9)
            
            # 从X中采样z和z'
            z = random.uniform(0, 2*PI)
            z_prime = random.uniform(0, 2*PI)
            
            T_sl = self.cert.T(x0, z, s, l)
            T_ll_prime = self.cert.T(z, z_prime, l, l_prime)
            T_sl_prime = self.cert.T(x0, z_prime, s, l_prime)
            
            # 检查前提条件
            if T_sl >= -self.tol and T_ll_prime >= -self.tol:
                # 检查下降条件
                if T_sl_prime + self.epsilon > T_sl + self.tol:
                    violations.append({
                        'x0': x0, 'z': z, 'z_prime': z_prime,
                        'T_sl': T_sl, 'T_ll_prime': T_ll_prime,
                        'T_sl_prime': T_sl_prime
                    })
        
        print(f"测试次数: {self.num_tests // 20}")
        print(f"违反次数: {len(violations)}")
        
        if violations:
            print("\n违反示例:")
            for i, v in enumerate(violations[:5]):
                print(f"  示例{i+1}: x0={v['x0']:.4f}, z={v['z']:.4f}, z'={v['z_prime']:.4f}")
                print(f"    T_{s}{l}(x0, z) = {v['T_sl']:.6f} >= 0")
                print(f"    T_{l}{l_prime}(z, z') = {v['T_ll_prime']:.6f} >= 0")
                print(f"    T_{s}{l_prime}(x0, z') = {v['T_sl_prime']:.6f}")
                print(f"    T_{s}{l_prime}(x0, z') + ε = {v['T_sl_prime'] + self.epsilon:.6f}")
                print(f"    ε = {self.epsilon:.6f}")
                print(f"    不满足: T_sl' + ε <= T_sl")
        
        return len(violations) == 0
    
    def test_automaton_simulation(self, num_steps=20):
        """模拟自动机运行，验证证书性质"""
        print("\n" + "="*60)
        print("模拟自动机运行")
        print("="*60)
        
        # 从X0中随机选择初始状态
        x0 = random.uniform(0, PI/9)
        x = x0
        q = Automaton.get_initial_state()  # q = 1
        
        print(f"初始系统状态: x0 = {x0:.4f}")
        print(f"初始自动机状态: q0 = {q}")
        print()
        
        accept_count = 0
        T_values = []
        
        for step in range(num_steps):
            x_next = f_m(x)
            L_x = Automaton.get_label(x)
            
            # 自动机转移
            q_next_list = Automaton.delta(q, L_x)
            q_next = q_next_list[0]  # 对于确定性自动机
            
            # 计算条件1的T值
            T_val = self.cert.T(x, x_next, q, q_next)
            T_values.append(T_val)
            
            print(f"步骤 {step}:")
            print(f"  x = {x:.4f}, f(x) = {x_next:.4f}")
            print(f"  L(x) = {L_x} (x ∈ Xu: {In_Unsafe(x)})")
            print(f"  自动机: {q} -> {q_next}")
            print(f"  T_{q}{q_next}(x, f(x)) = {T_val:.6f}")
            
            if q_next == 0:  # 进入接受状态
                accept_count += 1
                print(f"  ✓ 进入接受状态")
            
            # 检查条件1
            if T_val < -self.tol:
                print(f"  ⚠ 违反条件1: T < 0!")
            
            x = x_next
            q = q_next
        
        print(f"\n总结:")
        print(f"  总步数: {num_steps}")
        print(f"  进入接受状态次数: {accept_count}")
        print(f"  T值范围: [{min(T_values):.6f}, {max(T_values):.6f}]")
        
        # 检查下降条件
        if len(T_values) >= 2 and accept_count > 0:
            print("\n下降趋势分析:")
            for i in range(1, len(T_values)):
                if T_values[i] < T_values[i-1]:
                    print(f"  步骤{i}: 下降 {T_values[i-1] - T_values[i]:.6f}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("闭包证书验证测试")
        print("="*60)
        
        # 打印证书参数
        self._print_certificate_info()
        
        # 运行测试
        cond1_passed = self.test_condition1()
        cond2_passed = self.test_condition2()
        cond3_passed = self.test_condition3()
        
        # 模拟运行
        self.test_automaton_simulation(num_steps=10)
        
        # 总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"条件1 (相邻转移非负): {'通过 ✓' if cond1_passed else '失败 ✗'}")
        print(f"条件2 (迁移蕴含): {'通过 ✓' if cond2_passed else '失败 ✗'}")
        print(f"条件3 (下降条件): {'通过 ✓' if cond3_passed else '失败 ✗'}")
        
        all_passed = cond1_passed and cond2_passed and cond3_passed
        print(f"\n总体结果: {'所有条件通过! ✓' if all_passed else '存在失败! ✗'}")
        
        return all_passed
    
    def _print_certificate_info(self):
        """打印证书信息"""
        print(f"下降参数 ε = {self.epsilon:.6f}")
        print("\n系数:")
        
        names = ["T_00", "T_01", "T_10", "T_11"]
        for i, name in enumerate(names):
            base_idx = i * 5
            coeff_slice = self.coeffs[base_idx:base_idx+5]
            print(f"  {name}: {coeff_slice}")

# ==================== 主函数 ====================
def main():
    # 您的证书系数
    coeffs = [
        0.000000,   # c01
        0.000000,   # c11
        0.000000,   # c21
        0.000000,   # c31
        0.000000,   # c41
        
        -0.500000,  # c02
        0.000000,   # c12
        0.000000,   # c22
        0.000000,   # c32
        0.000000,   # c42
        
        -4.310550,  # c03
        1.764117,   # c13
        0.000000,   # c23
        1.764116,   # c33
        0.000000,   # c43
        
        0.000000,   # c04
        0.000000,   # c14
        0.000000,   # c24
        0.000000,   # c34
        0.000000,   # c44
    ]
    
    epsilon = 0.5
    
    # 设置随机种子以便重现结果
    random.seed(30)
    
    # 创建测试器并运行测试
    tester = ClosureCertificateTester(coeffs, epsilon, num_tests=10000)
    all_passed = tester.run_all_tests()
    
    # 额外边界测试
    test_boundary_cases(coeffs, epsilon)
    
    return all_passed

def test_boundary_cases(coeffs, epsilon):
    """测试边界情况"""
    print("\n" + "="*60)
    print("边界情况测试")
    print("="*60)
    
    cert = ClosureCertificate(coeffs)
    
    boundary_points = [
        (0.0, 0.0, "X0边界-X0边界"),
        (PI/9, PI/9, "X0边界-X0边界"),
        (7*PI/9, 7*PI/9, "Xu边界-Xu边界"),
        (8*PI/9, 8*PI/9, "Xu边界-Xu边界"),
        (0.0, 7*PI/9, "X0边界-Xu边界"),
        (7*PI/9, 0.0, "Xu边界-X0边界"),
    ]
    
    for x, y, desc in boundary_points:
        print(f"\n{desc}: x={x:.4f}, y={y:.4f}")
        
        for i in [0, 1]:
            for j in [0, 1]:
                T_val = cert.T(x, y, i, j)
                status = "✓" if T_val >= -TOL else "✗"
                print(f"  {status} T_{i}{j} = {T_val:.6f}")

if __name__ == "__main__":
    all_passed = main()
    
    if all_passed:
        print("\n" + "="*60)
        print("恭喜! 闭包证书验证通过!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("注意: 闭包证书验证失败!")
        print("="*60)