import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# ==================== 系统参数定义 ====================
TS = 0.1
OMEGA = 0.01
K = 0.0006
PI = 3.1415926

# ==================== 闭包证书参数 ====================
# 从您的合成结果获取的参数
epsilon = 0.000510

# T_00 系数
c00 = 0.003058
c10 = 0.000510
c20 = -0.000510

# T_01 系数
c01 = -0.000510
c11 = -0.000510
c21 = -1.227905

# T_10 系数
c02 = -0.023626
c12 = 0.010745
c22 = -0.000510

# T_11 系数
c03 = 100.000000
c13 = -0.000510
c23 = -46.250723

# 将系数组织为列表
coeffs = [c00, c10, c20,  # T_00
          c01, c11, c21,  # T_01
          c02, c12, c22,  # T_10
          c03, c13, c23]  # T_11

# ==================== 辅助函数定义 ====================
def f_m(x):
    """离散时间动力学函数"""
    return x + TS * OMEGA + TS * K * math.sin(-x) - 0.532 * x ** 2 + 1.69

def In_X0(x):
    """检查x是否在初始集中"""
    return x >= 0 and x <= PI / 9

def In_Unsafe(x):
    """检查x是否在不安全集中"""
    return x >= 7 * PI / 9 and x <= 8 * PI / 9

def delta(i, unsafe_flag):
    """
    模态转移函数
    i: 当前自动机状态 (0或1)
    unsafe_flag: True表示x在不安全集中
    返回: 可能的下一状态列表
    """
    if i == 1:
        if unsafe_flag:  # 如果x在不安全集中
            return [0]
        else:            # 如果x在安全集中
            return [1]
    elif i == 0:
        return [0]  # 状态0是吸收状态
    else:
        raise ValueError(f"无效的自动机状态: {i}")

def T_template(x, y, i, j, coeffs):
    """根据i,j计算对应的闭包证书值"""
    if i == 0 and j == 0:
        # T_00(x, y) = c00 + c10*x + c20*y
        return coeffs[0] + coeffs[1] * x + coeffs[2] * y
    elif i == 0 and j == 1:
        # T_01(x, y) = c01 + c11*x + c21*y
        return coeffs[3] + coeffs[4] * x + coeffs[5] * y
    elif i == 1 and j == 0:
        # T_10(x, y) = c02 + c12*x + c22*y
        return coeffs[6] + coeffs[7] * x + coeffs[8] * y
    elif i == 1 and j == 1:
        # T_11(x, y) = c03 + c13*x + c23*y
        return coeffs[9] + coeffs[10] * x + coeffs[11] * y
    else:
        raise ValueError(f"无效的状态对: ({i}, {j})")

# ==================== 测试函数定义 ====================
def test_condition1(system_states, automaton_states, verbose=False):
    """
    测试条件1: T_{i,i'}(x, x') >= 0
    其中i' ∈ δ(i, L(x)), x' = f(x)
    """
    violations = []
    total_tests = 0
    
    for idx in range(len(system_states)-1):
        x = system_states[idx]
        x_next = system_states[idx+1]
        i = automaton_states[idx]
        
        # 计算标签函数 L(x)
        L_x = In_Unsafe(x)
        
        # 获取可能的下一自动机状态
        i_prime_list = delta(i, L_x)
        
        for i_prime in i_prime_list:
            # 计算 T_{i,i'}(x, x')
            T_value = T_template(x, x_next, i, i_prime, coeffs)
            
            total_tests += 1
            
            if T_value < 0:
                violations.append({
                    'step': idx,
                    'x': x,
                    'x_next': x_next,
                    'i': i,
                    'i_prime': i_prime,
                    'T_value': T_value
                })
                
                if verbose:
                    print(f"条件1违反: 步骤{idx}, x={x:.6f}, x'={x_next:.6f}, "
                          f"i={i}, i'={i_prime}, T={T_value:.6f}")
    
    violation_rate = len(violations) / total_tests if total_tests > 0 else 0
    
    return {
        'total_tests': total_tests,
        'violations': violations,
        'violation_count': len(violations),
        'violation_rate': violation_rate
    }

def test_condition2(system_states, automaton_states, all_y_samples=100, verbose=False):
    """
    测试条件2: T_{i',j}(x', y) >= 0 ⇒ T_{i,j}(x, y) >= 0
    其中i' ∈ δ(i, L(x))
    """
    violations = []
    total_tests = 0
    
    # 从状态空间中采样y值
    y_samples = np.random.uniform(0, 2*PI, all_y_samples)
    
    for idx in range(len(system_states)-1):
        x = system_states[idx]
        x_next = system_states[idx+1]
        i = automaton_states[idx]
        
        # 计算标签函数 L(x)
        L_x = In_Unsafe(x)
        
        # 获取可能的下一自动机状态
        i_prime_list = delta(i, L_x)
        
        for y in y_samples:
            for j in [0, 1]:  # 对所有可能的j
                for i_prime in i_prime_list:
                    # 前提: T_{i',j}(x', y) >= 0
                    premise = T_template(x_next, y, i_prime, j, coeffs)
                    
                    # 结论: T_{i,j}(x, y) >= 0
                    conclusion = T_template(x, y, i, j, coeffs)
                    
                    total_tests += 1
                    
                    if premise >= 0 and conclusion < 0:
                        violations.append({
                            'step': idx,
                            'x': x,
                            'x_next': x_next,
                            'y': y,
                            'i': i,
                            'i_prime': i_prime,
                            'j': j,
                            'premise': premise,
                            'conclusion': conclusion
                        })
                        
                        if verbose:
                            print(f"条件2违反: 步骤{idx}, x={x:.6f}, x'={x_next:.6f}, y={y:.6f}, "
                                  f"i={i}, i'={i_prime}, j={j}")
                            print(f"  前提 T_{i_prime}{j}(x',y) = {premise:.6f}")
                            print(f"  结论 T_{i}{j}(x,y) = {conclusion:.6f}")
    
    violation_rate = len(violations) / total_tests if total_tests > 0 else 0
    
    return {
        'total_tests': total_tests,
        'violations': violations,
        'violation_count': len(violations),
        'violation_rate': violation_rate
    }

def test_condition3(initial_states, all_z_samples=50, all_z_prime_samples=50, verbose=False):
    """
    测试条件3: (T_{s,l}(x0, z) >= 0 and T_{l,l'}(z, z') >= 0) 
              ⇒ (T_{s,l'}(x0, z') + ε <= T_{s,l}(x0, z))
    其中s = q0 = 1, l, l' ∈ Acc = {0}
    """
    violations = []
    total_tests = 0
    
    s = 1  # 初始自动机状态
    l = 0  # 接受状态
    l_prime = 0  # 接受状态
    
    # 采样z和z'值
    z_samples = np.random.uniform(0, 2*PI, all_z_samples)
    z_prime_samples = np.random.uniform(0, 2*PI, all_z_prime_samples)
    
    for x0 in initial_states:
        for z in z_samples:
            for z_prime in z_prime_samples:
                # 前提1: T_{s,l}(x0, z) >= 0
                premise1 = T_template(x0, z, s, l, coeffs)
                
                # 前提2: T_{l,l'}(z, z') >= 0
                premise2 = T_template(z, z_prime, l, l_prime, coeffs)
                
                # 结论: T_{s,l'}(x0, z') + ε <= T_{s,l}(x0, z)
                left_side = T_template(x0, z_prime, s, l_prime, coeffs) + epsilon
                right_side = T_template(x0, z, s, l, coeffs)
                
                total_tests += 1
                
                if premise1 >= 0 and premise2 >= 0 and left_side > right_side:
                    violations.append({
                        'x0': x0,
                        'z': z,
                        'z_prime': z_prime,
                        'premise1': premise1,
                        'premise2': premise2,
                        'left_side': left_side,
                        'right_side': right_side,
                        'difference': left_side - right_side
                    })
                    
                    if verbose:
                        print(f"条件3违反: x0={x0:.6f}, z={z:.6f}, z'={z_prime:.6f}")
                        print(f"  T_{s}{l}(x0,z) = {premise1:.6f}")
                        print(f"  T_{l}{l_prime}(z,z') = {premise2:.6f}")
                        print(f"  T_{s}{l_prime}(x0,z') + ε = {left_side:.6f}")
                        print(f"  T_{s}{l}(x0,z) = {right_side:.6f}")
                        print(f"  差值 = {left_side - right_side:.6f}")
    
    violation_rate = len(violations) / total_tests if total_tests > 0 else 0
    
    return {
        'total_tests': total_tests,
        'violations': violations,
        'violation_count': len(violations),
        'violation_rate': violation_rate
    }

def simulate_system(initial_x, initial_q, steps=100):
    """
    模拟系统演化
    initial_x: 初始系统状态
    initial_q: 初始自动机状态
    steps: 模拟步数
    """
    system_states = [initial_x]
    automaton_states = [initial_q]
    
    current_x = initial_x
    current_q = initial_q
    
    for step in range(steps):
        # 计算下一系统状态
        next_x = f_m(current_x)
        
        # 计算标签函数
        unsafe_flag = In_Unsafe(current_x)
        
        # 计算下一自动机状态
        # 注意：delta可能返回多个状态，我们取第一个
        next_q_candidates = delta(current_q, unsafe_flag)
        next_q = next_q_candidates[0]  # 取第一个转移
        
        system_states.append(next_x)
        automaton_states.append(next_q)
        
        current_x = next_x
        current_q = next_q
    
    return system_states, automaton_states

def run_comprehensive_test(num_initial_points=1000, steps_per_trajectory=100, 
                          verbose_interval=100, save_plot=True):
    """
    运行综合测试
    """
    print("="*60)
    print("闭包证书数值验证测试")
    print("="*60)
    print(f"测试配置:")
    print(f"  - 初始点数量: {num_initial_points}")
    print(f"  - 每个轨迹步数: {steps_per_trajectory}")
    print(f"  - ε 参数: {epsilon}")
    print("="*60)
    
    # 生成初始点
    print("\n1. 生成初始点和模拟轨迹...")
    initial_points = []
    all_system_states = []
    all_automaton_states = []
    
    for i in tqdm(range(num_initial_points), desc="模拟轨迹"):
        # 从初始集X0中随机选择初始点
        initial_x = random.uniform(0, PI/9)
        initial_q = 1  # 根据您的描述，初始自动机状态为1
        
        initial_points.append(initial_x)
        
        # 模拟轨迹
        system_states, automaton_states = simulate_system(initial_x, initial_q, steps_per_trajectory)
        all_system_states.append(system_states)
        all_automaton_states.append(automaton_states)
    
    print(f"  生成了 {num_initial_points} 条轨迹，每条 {steps_per_trajectory} 步")
    
    # 测试条件1
    print("\n2. 测试条件1: T_{i,i'}(x, f(x)) >= 0")
    cond1_results_all = []
    cond1_violations_total = 0
    cond1_tests_total = 0
    
    for traj_idx in tqdm(range(num_initial_points), desc="测试条件1"):
        system_states = all_system_states[traj_idx]
        automaton_states = all_automaton_states[traj_idx]
        
        cond1_result = test_condition1(system_states, automaton_states, verbose=False)
        
        cond1_results_all.append(cond1_result)
        cond1_violations_total += cond1_result['violation_count']
        cond1_tests_total += cond1_result['total_tests']
        
        if (traj_idx + 1) % verbose_interval == 0:
            print(f"  已处理 {traj_idx+1} 条轨迹，违反次数: {cond1_violations_total}")
    
    cond1_violation_rate = cond1_violations_total / cond1_tests_total if cond1_tests_total > 0 else 0
    
    # 测试条件2
    print("\n3. 测试条件2: T_{i',j}(f(x), y) >= 0 ⇒ T_{i,j}(x, y) >= 0")
    cond2_results_all = []
    cond2_violations_total = 0
    cond2_tests_total = 0
    
    # 对每条轨迹测试条件2
    for traj_idx in tqdm(range(min(num_initial_points, 100)), desc="测试条件2"):
        system_states = all_system_states[traj_idx]
        automaton_states = all_automaton_states[traj_idx]
        
        # 对每条轨迹采样较少的y值以提高效率
        cond2_result = test_condition2(system_states, automaton_states, all_y_samples=20, verbose=False)
        
        cond2_results_all.append(cond2_result)
        cond2_violations_total += cond2_result['violation_count']
        cond2_tests_total += cond2_result['total_tests']
    
    cond2_violation_rate = cond2_violations_total / cond2_tests_total if cond2_tests_total > 0 else 0
    
    # 测试条件3
    print("\n4. 测试条件3: 下降条件")
    cond3_result = test_condition3(initial_points, all_z_samples=50, all_z_prime_samples=50, verbose=False)
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    print("\n条件1: T_{i,i'}(x, f(x)) >= 0")
    print(f"  总测试数: {cond1_tests_total:,}")
    print(f"  违反次数: {cond1_violations_total}")
    print(f"  违反率: {cond1_violation_rate*100:.6f}%")
    
    print("\n条件2: T_{i',j}(f(x), y) >= 0 ⇒ T_{i,j}(x, y) >= 0")
    print(f"  总测试数: {cond2_tests_total:,}")
    print(f"  违反次数: {cond2_violations_total}")
    print(f"  违反率: {cond2_violation_rate*100:.6f}%")
    
    print(f"\n条件3: 下降条件")
    print(f"  总测试数: {cond3_result['total_tests']:,}")
    print(f"  违反次数: {cond3_result['violation_count']}")
    print(f"  违反率: {cond3_result['violation_rate']*100:.6f}%")
    
    # 打印详细违反信息（如果有）
    if cond1_violations_total > 0 and len(cond1_results_all) > 0:
        print(f"\n条件1详细违反信息（前5个）:")
        violation_count = 0
        for traj_idx, result in enumerate(cond1_results_all):
            for violation in result['violations'][:5-violation_count]:
                print(f"  轨迹{traj_idx}, 步骤{violation['step']}: "
                      f"x={violation['x']:.6f}, x'={violation['x_next']:.6f}, "
                      f"i={violation['i']}, i'={violation['i_prime']}, "
                      f"T={violation['T_value']:.6f}")
                violation_count += 1
                if violation_count >= 5:
                    break
            if violation_count >= 5:
                break
    
    if cond2_violations_total > 0 and len(cond2_results_all) > 0:
        print(f"\n条件2详细违反信息（前3个）:")
        violation_count = 0
        for traj_idx, result in enumerate(cond2_results_all):
            for violation in result['violations'][:3-violation_count]:
                print(f"  轨迹{traj_idx}, 步骤{violation['step']}: "
                      f"x={violation['x']:.6f}, y={violation['y']:.6f}, "
                      f"i={violation['i']}, j={violation['j']}, "
                      f"i'={violation['i_prime']}")
                violation_count += 1
                if violation_count >= 3:
                    break
            if violation_count >= 3:
                break
    
    if cond3_result['violation_count'] > 0:
        print(f"\n条件3详细违反信息（前3个）:")
        for i, violation in enumerate(cond3_result['violations'][:3]):
            print(f"  测试{i+1}: x0={violation['x0']:.6f}, z={violation['z']:.6f}, z'={violation['z_prime']:.6f}")
            print(f"    T_sl(x0,z)={violation['premise1']:.6f}, "
                  f"T_ll'(z,z')={violation['premise2']:.6f}")
            print(f"    左边(T_sl'(x0,z')+ε)={violation['left_side']:.6f}, "
                  f"右边(T_sl(x0,z))={violation['right_side']:.6f}")
            print(f"    差值={violation['difference']:.6f}")
    
    # 创建可视化
    if save_plot:
        print("\n5. 生成可视化图表...")
        create_visualization(all_system_states, all_automaton_states, 
                            cond1_violation_rate, cond2_violation_rate, 
                            cond3_result['violation_rate'])
    
    # 检查整体结果
    all_passed = (cond1_violations_total == 0 and 
                  cond2_violations_total == 0 and 
                  cond3_result['violation_count'] == 0)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有条件都通过测试！")
    else:
        print("❌ 存在违反条件的情况")
    print("="*60)
    
    return {
        'cond1': {
            'violations': cond1_violations_total,
            'total_tests': cond1_tests_total,
            'violation_rate': cond1_violation_rate
        },
        'cond2': {
            'violations': cond2_violations_total,
            'total_tests': cond2_tests_total,
            'violation_rate': cond2_violation_rate
        },
        'cond3': {
            'violations': cond3_result['violation_count'],
            'total_tests': cond3_result['total_tests'],
            'violation_rate': cond3_result['violation_rate']
        },
        'all_passed': all_passed
    }

def create_visualization(all_system_states, all_automaton_states, 
                        cond1_violation_rate, cond2_violation_rate, 
                        cond3_violation_rate):
    """
    创建可视化图表
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 系统状态轨迹示例
    ax1 = axes[0, 0]
    for i in range(min(10, len(all_system_states))):
        ax1.plot(all_system_states[i], linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('步数')
    ax1.set_ylabel('系统状态 x')
    ax1.set_title('系统状态轨迹示例')
    ax1.grid(True, alpha=0.3)
    
    # 2. 自动机状态轨迹示例
    ax2 = axes[0, 1]
    for i in range(min(10, len(all_automaton_states))):
        ax2.plot(all_automaton_states[i], linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('步数')
    ax2.set_ylabel('自动机状态 q')
    ax2.set_title('自动机状态轨迹示例')
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 3. 初始状态分布
    ax3 = axes[0, 2]
    initial_states = [states[0] for states in all_system_states]
    ax3.hist(initial_states, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('初始状态 x0')
    ax3.set_ylabel('频数')
    ax3.set_title('初始状态分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 条件违反率
    ax4 = axes[1, 0]
    conditions = ['条件1', '条件2', '条件3']
    violation_rates = [cond1_violation_rate*100, cond2_violation_rate*100, cond3_violation_rate*100]
    bars = ax4.bar(conditions, violation_rates, color=['red' if rate > 0 else 'green' for rate in violation_rates])
    ax4.set_ylabel('违反率 (%)')
    ax4.set_title('各条件违反率')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, rate in zip(bars, violation_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.4f}%', ha='center', va='bottom', fontsize=9)
    
    # 5. 系统状态相图
    ax5 = axes[1, 1]
    for i in range(min(5, len(all_system_states))):
        x = all_system_states[i]
        x_next = x[1:] + [x[-1]]  # 简单的移位，用于相图
        ax5.scatter(x[:-1], x_next[:-1], s=1, alpha=0.5)
    ax5.set_xlabel('x(t)')
    ax5.set_ylabel('x(t+1)')
    ax5.set_title('系统状态相图')
    ax5.grid(True, alpha=0.3)
    
    # 6. 证书值分布示例
    ax6 = axes[1, 2]
    # 随机选择一些点计算证书值
    T_values = []
    for _ in range(1000):
        x = random.uniform(0, 2*PI)
        y = random.uniform(0, 2*PI)
        i = random.choice([0, 1])
        j = random.choice([0, 1])
        T_val = T_template(x, y, i, j, coeffs)
        T_values.append(T_val)
    
    ax6.hist(T_values, bins=50, edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=1, label='T=0')
    ax6.set_xlabel('T值')
    ax6.set_ylabel('频数')
    ax6.set_title('证书值T分布示例')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('closure_certificate_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  图表已保存为 'closure_certificate_test_results.png'")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 运行综合测试
    results = run_comprehensive_test(
        num_initial_points=1000,      # 1000个初始点
        steps_per_trajectory=100,     # 每个点执行100步
        verbose_interval=200,         # 每200条轨迹打印一次进度
        save_plot=True                # 保存可视化图表
    )
    
    # 保存结果到文件
    with open('test_results.txt', 'w') as f:
        f.write("闭包证书数值测试结果\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"初始点数量: 1000\n")
        f.write(f"每条轨迹步数: 100\n")
        f.write(f"ε 参数: {epsilon}\n")
        f.write("="*60 + "\n\n")
        
        f.write("条件1结果:\n")
        f.write(f"  总测试数: {results['cond1']['total_tests']:,}\n")
        f.write(f"  违反次数: {results['cond1']['violations']}\n")
        f.write(f"  违反率: {results['cond1']['violation_rate']*100:.6f}%\n\n")
        
        f.write("条件2结果:\n")
        f.write(f"  总测试数: {results['cond2']['total_tests']:,}\n")
        f.write(f"  违反次数: {results['cond2']['violations']}\n")
        f.write(f"  违反率: {results['cond2']['violation_rate']*100:.6f}%\n\n")
        
        f.write("条件3结果:\n")
        f.write(f"  总测试数: {results['cond3']['total_tests']:,}\n")
        f.write(f"  违反次数: {results['cond3']['violations']}\n")
        f.write(f"  违反率: {results['cond3']['violation_rate']*100:.6f}%\n\n")
        
        f.write("="*60 + "\n")
        if results['all_passed']:
            f.write("✅ 所有条件都通过测试！\n")
        else:
            f.write("❌ 存在违反条件的情况\n")
        f.write("="*60 + "\n")
    
    print("\n详细结果已保存到 'test_results.txt'")