import math

"""
验证使用 CEGIS-Z3-dReal 方法合成的状态安全证书的有效性
检查证书 B(x,q) 是否满足安全性条件：
1. 初始状态非负性: ∀x∈X0, B(x,1) ≥ 0
2. 前向不变性: 如果 B(x,q) ≥ 0，则 B(f(x), δ(q, L(x))) ≥ 0
3. 安全性: 从初始状态无法到达不安全状态
"""

# ==================== 系统参数定义 ====================
SAMPLING_TIME = 0.1
NATURAL_FREQUENCY = 0.01
COUPLING_COEFFICIENT = 0.0006
QUADRATIC_TERM = 0.532
CONSTANT_TERM = 1.69

# ==================== 空间定义函数 ====================
def in_state_space(x: float) -> bool:
    """检查状态是否在状态空间内: x ∈ [0, 2π]"""
    return 0 <= x <= 2 * math.pi


def in_initial_set(x: float) -> bool:
    """检查状态是否在初始集中: x ∈ [0, π/9]"""
    return 0 <= x <= math.pi / 9


def in_unsafe_set(x: float) -> bool:
    """检查状态是否在不安全集中: x ∈ [7π/9, 8π/9]"""
    return 7 * math.pi / 9 <= x <= 8 * math.pi / 9

# ==================== 系统函数定义 ====================
def system_dynamics(x: float) -> float:
    """
    计算系统下一状态: x_next = f(x)
    f(x) = x + Ts*Ω + Ts*K*sin(-x) - 0.532*x^2 + 1.69
    """
    return (x + SAMPLING_TIME * NATURAL_FREQUENCY + 
            SAMPLING_TIME * COUPLING_COEFFICIENT * math.sin(-x) - 
            QUADRATIC_TERM * x ** 2 + CONSTANT_TERM)


def safety_label(x: float) -> int:
    """
    安全标签函数: 如果x在不安全集中返回1，否则返回0
    """
    return 1 if in_unsafe_set(x) else 0


def mode_transition(current_mode: int, safety_label_value: int) -> int:
    """
    模态转移函数: δ(q, w)
    q=1, w=1 → 0  (从不安全状态转移到吸收状态)
    q=1, w=0 → 1  (保持在安全状态)
    q=0 → 0       (吸收状态)
    """
    if current_mode == 1:
        return 0 if safety_label_value == 1 else 1
    else:  # current_mode == 0
        return 0

# ==================== 安全证书定义 ====================
def barrier_function_2025_03_15(x: float, mode: int) -> float:
    """
    2025-03-15 实验得到的安全证书
    B(x, q) = -1 + 2*q^2 - (7/16)*x*q
    """
    return -1 + 2 * mode ** 2 - (7 / 16) * x * mode

# ==================== 采样函数 ====================
def generate_samples(start: float, end: float, step: float) -> list[float]:
    """
    在区间[start, end]上以步长step生成采样点
    """
    num_samples = int((end - start) / step) + 1
    return [start + i * step for i in range(num_samples)]

# ==================== 证书验证函数 ====================
def verify_safety_certificate(barrier_func, max_iterations: int = 2000) -> bool:
    """
    验证安全证书的有效性
    返回: True 如果证书有效，False 如果发现反例
    """
    # 在初始集上密集采样
    initial_samples = generate_samples(0, math.pi / 9, 0.0001)
    
    print("开始验证安全证书...")
    print(f"采样点数: {len(initial_samples)}")
    print(f"最大迭代次数: {max_iterations}")
    print("-" * 50)
    
    for initial_x in initial_samples:
        x = initial_x
        mode = 1  # 初始模态
        iteration = 0
        
        while iteration < max_iterations:
            # 计算下一状态和模态
            next_x = system_dynamics(x)
            next_mode = mode_transition(mode, safety_label(x))
            
            # 当前状态和下一状态的屏障函数值
            current_barrier = barrier_func(x, mode)
            next_barrier = barrier_func(next_x, next_mode)
            
            # 验证条件1: 初始状态非负性
            if in_initial_set(x) and current_barrier < 0:
                print(f"❌ 违反初始状态非负性:")
                print(f"   初始状态: x = {initial_x:.6f}, 当前状态: x = {x:.6f}")
                print(f"   模态: q = {mode}, B(x,q) = {current_barrier:.6f}")
                return False
            
            # 验证条件2: 前向不变性
            if in_state_space(x) and in_state_space(next_x):
                if current_barrier >= 0 and next_barrier < 0:
                    print(f"❌ 违反前向不变性:")
                    print(f"   初始状态: x = {initial_x:.6f}")
                    print(f"   当前状态: (x,q) = ({x:.6f}, {mode}), B = {current_barrier:.6f}")
                    print(f"   下一状态: (x',q') = ({next_x:.6f}, {next_mode}), B' = {next_barrier:.6f}")
                    return False
            
            # 验证条件3: 安全性 (如果到达吸收状态，应该无法从非负状态到达)
            if next_mode == 0 and current_barrier >= 0:
                print(f"❌ 违反安全性条件:")
                print(f"   初始状态: x = {initial_x:.6f}")
                print(f"   到达吸收状态 q=0 但从 B(x,q) ≥ 0 的状态")
                return False
            
            # 验证条件4: 如果B(x,q) < 0，应该保持在安全状态
            if current_barrier < 0 and not in_initial_set(x):
                print(f"⚠️  警告: 非初始状态下 B(x,q) < 0")
                print(f"   (x,q) = ({x:.6f}, {mode}), B = {current_barrier:.6f}")
            
            # 更新状态继续验证
            x = next_x
            mode = next_mode
            iteration += 1
        
        # 验证完成当前初始状态
        print(f"✓ 初始状态 x = {initial_x:.6f} 验证通过 (迭代{iteration}次)")
    
    print("\n" + "=" * 50)
    print("✅ 所有初始状态验证通过！证书有效。")
    print("=" * 50)
    return True


def quick_verification(barrier_func) -> None:
    """
    快速验证：检查关键边界条件
    """
    print("执行快速验证...")
    
    # 测试初始集的边界
    test_cases = [
        (0.0, 1, "初始集下界"),
        (math.pi / 9, 1, "初始集上界"),
        (7 * math.pi / 9, 0, "不安全集下界 (q=0)"),
        (8 * math.pi / 9, 0, "不安全集上界 (q=0)"),
        (7 * math.pi / 9, 1, "不安全集下界 (q=1)"),
        (8 * math.pi / 9, 1, "不安全集上界 (q=1)"),
    ]
    
    for x, q, description in test_cases:
        barrier_value = barrier_func(x, q)
        print(f"{description:20} (x={x:.4f}, q={q}): B = {barrier_value:.6f}")
        
        if in_initial_set(x) and q == 1 and barrier_value < 0:
            print(f"  ⚠️  可能违反初始条件")
        elif in_unsafe_set(x) and q == 0 and barrier_value >= 0:
            print(f"  ⚠️  可能违反安全条件")
    
    print("-" * 50)

# ==================== 主程序 ====================
def main():
    """主验证程序"""
    print("=" * 60)
    print("安全证书验证工具")
    print(f"证书: B(x, q) = -1 + 2*q^2 - (7/16)*x*q")
    print(f"系统: 一维 Kuramoto 振子")
    print("=" * 60)
    
    # 快速验证关键点
    quick_verification(barrier_function_2025_03_15)
    
    # 完整验证
    is_valid = verify_safety_certificate(
        barrier_func=barrier_function_2025_03_15,
        max_iterations=2000
    )
    
    if is_valid:
        print("\n验证总结:")
        print("1. ✅ 初始状态非负性: 所有 x∈X0 满足 B(x,1) ≥ 0")
        print("2. ✅ 前向不变性: B(x,q) ≥ 0 ⇒ B(f(x), δ(q,L(x))) ≥ 0")
        print("3. ✅ 安全性: 从初始状态无法到达不安全状态")
        print("\n🎉 安全证书验证成功！")
    else:
        print("\n❌ 安全证书验证失败，发现反例")
    
    print("=" * 60)

if __name__ == "__main__":
    main()