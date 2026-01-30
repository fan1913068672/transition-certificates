import math
import z3
import dreal
import time

# ==================== 系统参数定义 ====================
TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926

# ==================== 集合定义 ====================
def In_X_Cond(x):
    return dreal.And(x >= 0, x <= PI * 2)

def In_X0_Cond(x):
    return dreal.And(x >= PI * 4 / 9, x <= PI * 5 / 9)

def In_Unsafe_Cond(x):
    return dreal.And(x >= 7 * PI / 9, x <= 8 * PI / 9)

def In_X0(x):
    return x >= PI * 4 / 9 and x <= PI * 5 / 9

def In_Unsafe(x):
    return x >= 7 * PI / 9 and x <= 8 * PI / 9

# ==================== 系统动力学 ====================
def f_t(x):
    return x + TS * OMEGA + TS * K * dreal.sin(-x) - 0.532 * x ** 2 + 1.69

def f_m(x):
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69

# ==================== 辅助函数 ====================
def step_sample(a, b, s):
    res = []
    for i in range(int(a * int(1 / s)), int(b * int(1 / s)) + 1):
        res.append(i * s)
    return res

def t2float(a, precision=14):
    s = a.as_decimal(precision)
    if s[-1] == '?':
        s = s[:-1]
    return float(s)

def region_type(x):
    if In_X0(x):
        return 0
    elif In_Unsafe(x):
        return 1
    else:
        return 2

# ==================== 闭包证书模板定义 ====================
class ClosureCertificate:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        
    def _base_template(self, x, y, base_idx, x_region, y_region):
        """基础模板，被其他模板调用"""
        c0, cx0, cy0, cxu, cyu = self.coeffs[base_idx:base_idx+5]
        
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
    
    def T(self, x, y, i, j):
        """计算T_ij(x, y)"""
        x_region = region_type(x) if isinstance(x, (int, float)) else None
        y_region = region_type(y) if isinstance(y, (int, float)) else None
        
        if i == 0 and j == 0:
            return self._base_template(x, y, 0, x_region, y_region)
        elif i == 0 and j == 1:
            return self._base_template(x, y, 5, x_region, y_region)
        elif i == 1 and j == 0:
            return self._base_template(x, y, 10, x_region, y_region)
        elif i == 1 and j == 1:
            return self._base_template(x, y, 15, x_region, y_region)
        else:
            raise ValueError(f"无效状态对: ({i}, {j})")

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
    def get_transitions():
        """返回所有可能的转移"""
        transitions = []
        for i in [0, 1]:
            for L_x in [True, False]:  # x在Xu中或不在
                i_prime_list = Automaton.delta(i, L_x)
                for i_prime in i_prime_list:
                    transitions.append((i, L_x, i_prime))
        return transitions

# ==================== dreal版本 ====================
class DRealCertificate(ClosureCertificate):
    def __init__(self, coeffs):
        super().__init__(coeffs)
    
    def T_dreal(self, x, y, i, j):
        """dreal版本的T函数"""
        x_x0, x_xu, x_other = self._region_type_dreal(x)
        y_x0, y_xu, y_other = self._region_type_dreal(y)
        
        # 构建分支选择
        return dreal.if_then_else(
            x_x0,
            self._build_y_branch(y_x0, y_xu, y_other, x, y, i, j, 0),
            dreal.if_then_else(
                x_xu,
                self._build_y_branch(y_x0, y_xu, y_other, x, y, i, j, 1),
                self._build_y_branch(y_x0, y_xu, y_other, x, y, i, j, 2)
            )
        )
    
    def _region_type_dreal(self, x):
        """dreal版本的区域判断"""
        in_x0 = In_X0_Cond(x)
        in_xu = In_Unsafe_Cond(x)
        return in_x0, in_xu, dreal.And(dreal.Not(in_x0), dreal.Not(in_xu))
    
    def _build_y_branch(self, y_x0, y_xu, y_other, x, y, i, j, x_region):
        """构建y分支的选择"""
        return dreal.if_then_else(
            y_x0,
            self._base_template(x, y, self._get_base_idx(i, j), x_region, 0),
            dreal.if_then_else(
                y_xu,
                self._base_template(x, y, self._get_base_idx(i, j), x_region, 1),
                self._base_template(x, y, self._get_base_idx(i, j), x_region, 2)
            )
        )
    
    def _get_base_idx(self, i, j):
        """根据i,j获取基础索引"""
        if i == 0 and j == 0:
            return 0
        elif i == 0 and j == 1:
            return 5
        elif i == 1 and j == 0:
            return 10
        elif i == 1 and j == 1:
            return 15
        else:
            raise ValueError(f"无效状态对: ({i}, {j})")

# ==================== 反例检查器 ====================
class CounterexampleChecker:
    def __init__(self, coeffs, epsilon, tolerance=1e-10, precision=1e-6):
        self.coeffs = coeffs
        self.epsilon = epsilon
        self.tol = tolerance
        self.precision = precision
        self.cert = DRealCertificate(coeffs)
        
    def check_condition1(self):
        """检查条件1的反例"""
        print("检查条件1...")
        
        for i, L_x, i_prime in Automaton.get_transitions():
            solver = dreal.Context()
            solver.SetLogic(dreal.Logic.QF_NRA)
            solver.config.precision = self.precision
            
            x = dreal.Variable('x')
            solver.DeclareVariable(x, 0, 2 * PI)
            solver.Assert(In_X_Cond(x))
            
            # 根据转移条件添加约束
            if L_x:
                solver.Assert(In_Unsafe_Cond(x))
            else:
                solver.Assert(dreal.Not(In_Unsafe_Cond(x)))
            
            # 检查T < 0的情况
            T_val = self.cert.T_dreal(x, f_t(x), i, i_prime)
            solver.Assert(T_val < -self.tol)
            
            model = solver.CheckSat()
            if model is not None:
                return self._extract_ce1(model, x, i, i_prime, L_x)
        
        return None
    
    def check_condition2(self):
        """检查条件2的反例"""
        print("检查条件2...")
        
        for i in [0, 1]:
            for j in [0, 1]:
                for L_x in [True, False]:
                    i_prime_list = Automaton.delta(i, L_x)
                    
                    for i_prime in i_prime_list:
                        ce = self._check_condition2_case(i, j, L_x, i_prime)
                        if ce is not None:
                            return ce
        
        return None
    
    def _check_condition2_case(self, i, j, L_x, i_prime):
        """检查条件2的具体情况"""
        solver = dreal.Context()
        solver.SetLogic(dreal.Logic.QF_NRA)
        solver.config.precision = self.precision
        
        x = dreal.Variable('x')
        y = dreal.Variable('y')
        solver.DeclareVariable(x, 0, 2*PI)
        solver.DeclareVariable(y, 0, 2*PI)
        
        solver.Assert(In_X_Cond(x))
        solver.Assert(In_X_Cond(y))
        
        if L_x:
            solver.Assert(In_Unsafe_Cond(x))
        else:
            solver.Assert(dreal.Not(In_Unsafe_Cond(x)))
        
        premise = self.cert.T_dreal(f_t(x), y, i_prime, j) >= 0
        conclusion = self.cert.T_dreal(x, y, i, j) < -self.tol
        solver.Assert(dreal.And(premise, conclusion))
        
        model = solver.CheckSat()
        if model is not None:
            return self._extract_ce2(model, x, y, i, j, i_prime, L_x)
        
        return None
    
    def check_condition3(self):
        """检查条件3的反例"""
        print("检查条件3...")
        
        solver = dreal.Context()
        solver.SetLogic(dreal.Logic.QF_NRA)
        solver.config.precision = self.precision
        
        x0 = dreal.Variable('x0')
        z = dreal.Variable('z')
        z_prime = dreal.Variable('z_prime')
        
        solver.DeclareVariable(x0, 0, PI/9)
        solver.DeclareVariable(z, 0, 2*PI)
        solver.DeclareVariable(z_prime, 0, 2*PI)
        
        s_val, l_val, l_prime_val = 1, 0, 0
        
        premise1 = self.cert.T_dreal(x0, z, s_val, l_val) >= 0
        premise2 = self.cert.T_dreal(z, z_prime, l_val, l_prime_val) >= 0
        conclusion_neg = (self.cert.T_dreal(x0, z_prime, s_val, l_prime_val) + 
                         self.epsilon - self.tol > 
                         self.cert.T_dreal(x0, z, s_val, l_val))
        
        solver.Assert(dreal.And(premise1, premise2, conclusion_neg))
        
        model = solver.CheckSat()
        if model is not None:
            return self._extract_ce3(model, x0, z, z_prime, s_val, l_val, l_prime_val)
        
        return None
    
    def _extract_ce1(self, model, x_var, i, i_prime, L_x):
        """提取条件1的反例"""
        x_val = model[x_var].mid()
        f_x_val = f_m(x_val)
        T_actual = self.cert._base_template(x_val, f_x_val, 
                                           self.cert._get_base_idx(i, i_prime), 
                                           region_type(x_val), region_type(f_x_val))
        
        condition = "x in Xu" if L_x else "x not in Xu"
        print(f"发现条件1反例 (i={i}, i'={i_prime}, {condition}):")
        print(f"  x={x_val:.6f}, f(x)={f_x_val:.6f}")
        print(f"  T_{i}{i_prime}(x, f(x))={T_actual:.6f}")
        
        return ('condition1', i, i_prime, x_val, f_x_val)
    
    def _extract_ce2(self, model, x_var, y_var, i, j, i_prime, L_x):
        """提取条件2的反例"""
        x_val = model[x_var].mid()
        y_val = model[y_var].mid()
        f_x_val = f_m(x_val)
        
        cert = ClosureCertificate(self.coeffs)
        T_i_prime_j = cert.T(f_x_val, y_val, i_prime, j)
        T_i_j = cert.T(x_val, y_val, i, j)
        
        condition = "x in Xu" if L_x else "x not in Xu"
        print(f"发现条件2反例 (i={i}, j={j}, i'={i_prime}, {condition}):")
        print(f"  x={x_val:.6f}, f(x)={f_x_val:.6f}, y={y_val:.6f}")
        print(f"  T_{i_prime}{j}(f(x), y)={T_i_prime_j:.6f}")
        print(f"  T_{i}{j}(x, y)={T_i_j:.6f}")
        
        return ('condition2', i, j, i_prime, x_val, y_val)
    
    def _extract_ce3(self, model, x0_var, z_var, z_prime_var, s_val, l_val, l_prime_val):
        """提取条件3的反例"""
        x0_val = model[x0_var].mid()
        z_val = model[z_var].mid()
        z_prime_val = model[z_prime_var].mid()
        
        cert = ClosureCertificate(self.coeffs)
        T_sl = cert.T(x0_val, z_val, s_val, l_val)
        T_ll = cert.T(z_val, z_prime_val, l_val, l_prime_val)
        T_sl_prime = cert.T(x0_val, z_prime_val, s_val, l_prime_val)
        
        print("发现条件3反例:")
        print(f"  x0={x0_val:.6f}, z={z_val:.6f}, z'={z_prime_val:.6f}")
        print(f"  T_{s_val}{l_val}(x0, z)={T_sl:.6f}")
        print(f"  T_{l_val}{l_prime_val}(z, z')={T_ll:.6f}")
        print(f"  T_{s_val}{l_prime_val}(x0, z')={T_sl_prime:.6f}")
        print(f"  ε={self.epsilon:.6f}")
        print(f"  T_sl' + ε = {T_sl_prime + self.epsilon:.6f}")
        
        return ('condition3', s_val, l_val, l_prime_val, x0_val, z_val, z_prime_val)

# ==================== 主合成器 ====================
class ClosureCertificateSynthesizer:
    def __init__(self, max_iter=10000, tol=1e-6, precision=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.precision = precision
        
    def synthesize(self):
        """合成闭包证书"""
        print("开始合成闭包证书...")
        
        # 定义变量
        coeffs = [z3.Real(f'c{i:02d}') for i in range(20)]
        EPSILON = z3.Real('EPSILON')
        solver = z3.SolverFor("QF_NRA")
        
        # 约束范围
        for c in coeffs:
            solver.add(c >= -1000)
            solver.add(c <= 1000)
        solver.add(EPSILON > 0)
        solver.add(EPSILON <= 100)
        
        # 采样点
        X_samples = step_sample(0, 2.0 * PI, 0.1)
        Y_samples = X_samples.copy()
        X0_samples = step_sample(PI * 4 / 9, PI * 5 / 9, 0.1)
        
        # 添加初始约束
        self._add_initial_constraints(solver, coeffs, EPSILON, 
                                     X_samples, Y_samples, X0_samples)
        
        # CEGIS循环
        for iter_count in range(self.max_iter):
            if solver.check() != z3.sat:
                print("无法合成闭包证书")
                return None, None
            
            m = solver.model()
            coeffs_vals = [t2float(m[c]) for c in coeffs]
            epsilon_val = t2float(m[EPSILON])
            
            print(f"\n迭代 #{iter_count}:")
            print(f"ε = {epsilon_val:.6f}")
            for i, val in enumerate(coeffs_vals):
                print(f"c{i:02d} = {val:.6f}")
            
            # 检查反例
            checker = CounterexampleChecker(coeffs_vals, epsilon_val, self.tol, self.precision)
            
            ce_found = False
            for check_func in [checker.check_condition1, 
                             checker.check_condition2, 
                             checker.check_condition3]:
                ce = check_func()
                if ce is not None:
                    self._add_counterexample(solver, coeffs, EPSILON, ce)
                    ce_found = True
                    break
            
            if not ce_found:
                print(f"\n成功合成闭包证书，迭代次数: {iter_count}")
                return coeffs_vals, epsilon_val
        
        print(f"超过最大迭代次数 {self.max_iter}")
        return None, None
    
    def _add_initial_constraints(self, solver, coeffs, EPSILON, 
                               X_samples, Y_samples, X0_samples):
        """添加初始约束"""
        cert = ClosureCertificate(coeffs)
        
        # 条件1
        print("添加条件1约束...")
        for x in X_samples:
            xp = f_m(x)
            for i in [0, 1]:
                L_x = In_Unsafe(x)
                for i_prime in Automaton.delta(i, L_x):
                    T_val = cert.T(x, xp, i, i_prime)
                    solver.add(T_val >= 0)
        
        # 条件2
        print("添加条件2约束...")
        for x in X_samples[:10]:  # 减少样本数量
            xp = f_m(x)
            for y in Y_samples[:10]:
                for i in [0, 1]:
                    for j in [0, 1]:
                        L_x = In_Unsafe(x)
                        for i_prime in Automaton.delta(i, L_x):
                            premise = cert.T(xp, y, i_prime, j) >= 0
                            conclusion = cert.T(x, y, i, j) >= 0
                            solver.add(z3.Implies(premise, conclusion))
        
        # 条件3
        print("添加条件3约束...")
        s_val, l_val, l_prime_val = 1, 0, 0
        
        for x0 in X0_samples[:3]:
            for z in X_samples[:3]:
                for z_prime in X_samples[:3]:
                    premise1 = cert.T(x0, z, s_val, l_val) >= 0
                    premise2 = cert.T(z, z_prime, l_val, l_prime_val) >= 0
                    conclusion = cert.T(x0, z_prime, s_val, l_prime_val) + EPSILON <= cert.T(x0, z, s_val, l_val)
                    solver.add(z3.Implies(z3.And(premise1, premise2), conclusion))
    
    def _add_counterexample(self, solver, coeffs, EPSILON, ce):
        """添加反例到约束"""
        ce_type = ce[0]
        
        if ce_type == 'condition1':
            _, i, i_prime, x_val, f_x_val = ce
            cert = ClosureCertificate(coeffs)
            solver.add(cert.T(x_val, f_x_val, i, i_prime) >= 0)
            
        elif ce_type == 'condition2':
            _, i, j, i_prime, x_val, y_val = ce
            cert = ClosureCertificate(coeffs)
            premise = cert.T(f_m(x_val), y_val, i_prime, j) >= 0
            conclusion = cert.T(x_val, y_val, i, j) >= 0
            solver.add(z3.Implies(premise, conclusion))
            
        elif ce_type == 'condition3':
            _, s_val, l_val, l_prime_val, x0_val, z_val, z_prime_val = ce
            cert = ClosureCertificate(coeffs)
            premise1 = cert.T(x0_val, z_val, s_val, l_val) >= 0
            premise2 = cert.T(z_val, z_prime_val, l_val, l_prime_val) >= 0
            conclusion = cert.T(x0_val, z_prime_val, s_val, l_prime_val) + EPSILON <= cert.T(x0_val, z_val, s_val, l_val)
            solver.add(z3.Implies(z3.And(premise1, premise2), conclusion))

# ==================== 主程序 ====================
def main():
    start_time = time.time()
    
    synthesizer = ClosureCertificateSynthesizer(max_iter=1000)
    coeffs, epsilon_val = synthesizer.synthesize()
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f}秒")
    
    if coeffs is not None and epsilon_val is not None:
        print_results(coeffs, epsilon_val)
    else:
        print("\n闭包证书合成失败!")

def print_results(coeffs, epsilon):
    """打印结果"""
    print("\n" + "="*60)
    print("闭包证书合成成功!")
    print("="*60)
    
    print(f"\n下降参数 ε = {epsilon:.6f}")
    
    # 打印系数
    names = ["T_00", "T_01", "T_10", "T_11"]
    for i, name in enumerate(names):
        base_idx = i * 5
        print(f"\n{name}(x, y) 系数:")
        print(f"  c{base_idx:02d} = {coeffs[base_idx]:.6f}")
        print(f"  c{base_idx+1:02d} = {coeffs[base_idx+1]:.6f}")
        print(f"  c{base_idx+2:02d} = {coeffs[base_idx+2]:.6f}")
        print(f"  c{base_idx+3:02d} = {coeffs[base_idx+3]:.6f}")
        print(f"  c{base_idx+4:02d} = {coeffs[base_idx+4]:.6f}")
    
    # 测试点
    test_specific_points(coeffs, epsilon)

def test_specific_points(coeffs, epsilon):
    """测试特定点"""
    cert = ClosureCertificate(coeffs)
    
    test_points = [
        (0.1, 0.2, "X0-X0"),
        (0.1, 2.5, "X0-X_other"),
        (2.5, 0.1, "X_other-X0"),
        (7.8, 7.9, "Xu-Xu"),
        (2.5, 2.7, "X_other-X_other"),
    ]
    
    print("\n" + "="*60)
    print("测试点验证:")
    print("="*60)
    
    for x, y, desc in test_points:
        print(f"\n测试点 (x={x:.3f}, y={y:.3f}, 区域: {desc}):")
        for i in [0, 1]:
            for j in [0, 1]:
                T_val = cert.T(x, y, i, j)
                x_region = region_type(x)
                y_region = region_type(y)
                region_names = ["X0", "Xu", "X_other"]
                print(f"  T_{i}{j} = {T_val:.6f}  (x∈{region_names[x_region]}, y∈{region_names[y_region]})")

if __name__ == "__main__":
    main()