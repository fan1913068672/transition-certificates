import numpy as np
import torch
import math
import random
from typing import List, Tuple, Optional

# 神经网络定义
class BarrierNetwork(torch.nn.Module):
    """Shallow neural network for barrier certificate B(x1, x2, q)"""
    def __init__(self, input_dim=3, hidden_dim=10):
        super(BarrierNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, 1, bias=True)
        
    def forward(self, x1, x2, q):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)
            
        if x1.dim() == 0:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 0:
            x2 = x2.unsqueeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)
            
        inp = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1), q.unsqueeze(-1)], dim=-1)
        h = self.fc1(inp)  # Linear activation for analytical expression
        out = self.fc2(h)
        return out.squeeze(-1)

# 加载训练好的参数
def create_barrier_network():
    """创建并加载训练好的神经网络参数"""
    network = BarrierNetwork(input_dim=3, hidden_dim=10)
    
    # 第1层权重 (10x3)
    fc1_weight = torch.tensor([
        [ 0.22206941, -0.12351108,  0.30758172],
        [-0.36821434, -0.05034344, -0.03983097],
        [-0.2217223,  -0.292574,    0.02168754],
        [-0.20217259,  0.08935091, -0.01229939],
        [ 0.52530736, -0.3361786,  -0.51834047],
        [ 0.03101647, -0.3182618,   0.2595669 ],
        [-0.16042463,  0.06194159, -0.48793274],
        [ 0.09938865,  0.09659853,  0.31831124],
        [-0.29974195,  0.08115318, -0.45691633],
        [-0.33393818,  0.27361843, -0.5065321 ]
    ], dtype=torch.float32)
    
    # 第1层偏置 (10,)
    fc1_bias = torch.tensor([
        0.23029897, -0.38311592,  0.06139673,  0.6234876,  -0.17816922,  0.47668895,
        -0.08459653,  0.1782538,   0.23350956,  0.20422843
    ], dtype=torch.float32)
    
    # 第2层权重 (1x10)
    fc2_weight = torch.tensor([[
        -0.07332148,  0.14341801, -0.08049864, -0.12944159, -0.12501173, -0.03754412,
        -0.09970848, -0.00539517, -0.13214779, -0.14779091
    ]], dtype=torch.float32)
    
    # 第2层偏置 (1,)
    fc2_bias = torch.tensor([0.10287201], dtype=torch.float32)
    
    # 设置网络参数
    network.fc1.weight.data = fc1_weight
    network.fc1.bias.data = fc1_bias
    network.fc2.weight.data = fc2_weight
    network.fc2.bias.data = fc2_bias
    
    return network

# 系统定义
def f_m(x1, x2):
    """Dynamics in math format"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    x1p = x1 + Ts * Omega + 1.69 + Ts * K * math.sin(x2 - x1) - 0.532 * x1 ** 2
    x2p = x2 + Ts * Omega + 1.69 + Ts * K * math.sin(x1 - x2) - 0.532 * x2 ** 2
    return x1p, x2p

def In_X0(x1, x2):
    """Initial state constraint (Python version)"""
    return x1 >= 0 and x1 <= math.pi / 9 and x2 >= 0 and x2 <= math.pi / 9

def In_X(x1, x2):
    """State space constraint"""
    return x1 >= 0 and x1 <= 8 * math.pi / 9 and x2 >= 0 and x2 <= 8 * math.pi / 9

def In_Unsafe(x1, x2):
    """Unsafe set definition"""
    return (x1 >= 5/6 * math.pi and x1 <= 8/9 * math.pi) or \
           (x2 >= 5/6 * math.pi and x2 <= 8/9 * math.pi)

def q_trans(q, x1, x2):
    """Automaton state transitions"""
    if q == 1:
        if In_Unsafe(x1, x2):
            return 0
        else:
            return 1
    else:  # q == 0
        return 0

# 数值验证程序
def numerical_verification(n_samples=1000, n_steps=10, seed=42):
    """
    对屏障证书进行数值验证
    
    Args:
        n_samples: 初始状态采样点数
        n_steps: 每个点模拟的步数
        seed: 随机种子
    
    Returns:
        dict: 验证结果统计
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建神经网络
    barrier_net = create_barrier_network()
    barrier_net.eval()  # 设置为评估模式
    
    # 记录统计信息
    stats = {
        'total_samples': 0,
        'condition1_violations': 0,  # 初始状态非负
        'condition2_violations': 0,  # 不安全状态为负
        'condition3_violations': 0,  # 迁移非负 (q=1)
        'condition4_violations': 0,  # 迁移非负 (q=0)
        'outside_domain': 0,  # 超出状态空间的点
        'counterexamples': {
            'condition1': [],
            'condition2': [],
            'condition3': [],
            'condition4': []
        }
    }
    
    # 条件1: 从初始集合采样
    print(f"验证条件1: 初始状态非负 (B(x1, x2, 1) >= 0)")
    for _ in range(n_samples):
        # 在初始集合内均匀采样
        x1 = random.uniform(0, math.pi / 9)
        x2 = random.uniform(0, math.pi / 9)
        q = 1.0
        
        # 计算屏障值
        B_val = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
        
        # 检查条件1: B(x1, x2, 1) >= 0
        if B_val < 0:
            stats['condition1_violations'] += 1
            stats['counterexamples']['condition1'].append({
                'x1': x1, 'x2': x2, 'q': q, 'B': B_val
            })
    
    print(f"  采样 {n_samples} 个点，违反点数: {stats['condition1_violations']}")
    
    # 条件2: 在不安全集合内采样
    print(f"\n验证条件2: 不安全状态为负 (B(x1, x2, 0) < 0)")
    unsafe_samples = 0
    
    # 在状态空间内均匀采样，但只检查不安全区域内的点
    for _ in range(n_samples * 2):  # 采样更多点以确保覆盖不安全区域
        x1 = random.uniform(0, 8 * math.pi / 9)
        x2 = random.uniform(0, 8 * math.pi / 9)
        
        if In_Unsafe(x1, x2):
            unsafe_samples += 1
            q = 0.0
            
            # 计算屏障值
            B_val = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
            
            # 检查条件2: B(x1, x2, 0) < 0
            if B_val >= 0:
                stats['condition2_violations'] += 1
                stats['counterexamples']['condition2'].append({
                    'x1': x1, 'x2': x2, 'q': q, 'B': B_val
                })
        
        if unsafe_samples >= n_samples:
            break
    
    print(f"  找到 {unsafe_samples} 个不安全点，违反点数: {stats['condition2_violations']}")
    
    # 条件3和4: 模拟系统动态
    print(f"\n验证条件3&4: 迁移非负性")
    
    for _ in range(n_samples):
        # 在状态空间内均匀采样初始点
        x1 = random.uniform(0, 8 * math.pi / 9)
        x2 = random.uniform(0, 8 * math.pi / 9)
        
        # 确定初始自动机状态
        if In_Unsafe(x1, x2):
            q = 0.0
        else:
            q = 1.0
        
        # 模拟n_steps步
        for step in range(n_steps):
            # 检查当前点是否在状态空间内
            if not In_X(x1, x2):
                stats['outside_domain'] += 1
                break
            
            # 计算当前屏障值
            B_current = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
            
            # 计算下一个状态
            x1_next, x2_next = f_m(x1, x2)
            
            # 确定下一个自动机状态
            q_next = q_trans(q, x1, x2)
            
            # 计算下一个屏障值
            B_next = barrier_net(torch.tensor(x1_next), torch.tensor(x2_next), torch.tensor(float(q_next))).item()
            
            # 检查迁移条件: 如果当前B>=0，则下一个B>=0
            if B_current >= 0 and B_next < 0:
                if q == 1:
                    stats['condition3_violations'] += 1
                    stats['counterexamples']['condition3'].append({
                        'step': step,
                        'current': (x1, x2, q, B_current),
                        'next': (x1_next, x2_next, q_next, B_next)
                    })
                else:  # q == 0
                    stats['condition4_violations'] += 1
                    stats['counterexamples']['condition4'].append({
                        'step': step,
                        'current': (x1, x2, q, B_current),
                        'next': (x1_next, x2_next, q_next, B_next)
                    })
            
            # 更新状态
            x1, x2, q = x1_next, x2_next, q_next
    
    print(f"  模拟 {n_samples} 个轨迹，每个 {n_steps} 步")
    print(f"  q=1 迁移违反次数: {stats['condition3_violations']}")
    print(f"  q=0 迁移违反次数: {stats['condition4_violations']}")
    print(f"  超出状态空间轨迹数: {stats['outside_domain']}")
    
    # 总结
    print(f"\n{'='*50}")
    print("验证结果总结:")
    print(f"{'='*50}")
    
    total_violations = (stats['condition1_violations'] + 
                       stats['condition2_violations'] + 
                       stats['condition3_violations'] + 
                       stats['condition4_violations'])
    
    if total_violations == 0:
        print("✓ 所有验证条件均满足!")
    else:
        print(f"✗ 发现 {total_violations} 个违反:")
        if stats['condition1_violations'] > 0:
            print(f"  - 条件1违反: {stats['condition1_violations']} 个")
            print(f"    示例: {stats['counterexamples']['condition1'][:2]}")
        if stats['condition2_violations'] > 0:
            print(f"  - 条件2违反: {stats['condition2_violations']} 个")
            print(f"    示例: {stats['counterexamples']['condition2'][:2]}")
        if stats['condition3_violations'] > 0:
            print(f"  - 条件3违反: {stats['condition3_violations']} 个")
            print(f"    示例: {stats['counterexamples']['condition3'][:2]}")
        if stats['condition4_violations'] > 0:
            print(f"  - 条件4违反: {stats['condition4_violations']} 个")
            print(f"    示例: {stats['counterexamples']['condition4'][:2]}")
    
    return stats

# 额外的详细验证函数
def detailed_verification_from_initial(n_trajectories=100, max_steps=20):
    """从初始集合开始，详细验证系统演化"""
    barrier_net = create_barrier_network()
    barrier_net.eval()
    
    print("从初始集合开始详细验证:")
    print("-" * 50)
    
    violations = []
    
    for traj_idx in range(n_trajectories):
        # 在初始集合内采样
        x1 = random.uniform(0, math.pi / 9)
        x2 = random.uniform(0, math.pi / 9)
        q = 1.0
        
        trajectory_info = {
            'id': traj_idx,
            'initial': (x1, x2, q),
            'steps': [],
            'condition1_ok': True,
            'all_migrations_ok': True
        }
        
        # 检查初始条件
        B_init = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
        if B_init < 0:
            trajectory_info['condition1_ok'] = False
            trajectory_info['B_init'] = B_init
        
        # 模拟系统演化
        for step in range(max_steps):
            if not In_X(x1, x2):
                trajectory_info['steps'].append(f"Step {step}: 超出状态空间")
                break
            
            B_current = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
            x1_next, x2_next = f_m(x1, x2)
            q_next = q_trans(q, x1, x2)
            B_next = barrier_net(torch.tensor(x1_next), torch.tensor(x2_next), torch.tensor(float(q_next))).item()
            
            # 检查迁移条件
            if B_current >= 0 and B_next < 0:
                trajectory_info['all_migrations_ok'] = False
                trajectory_info['steps'].append(
                    f"Step {step}: 迁移违反! B({x1:.3f},{x2:.3f},{q})={B_current:.6f} -> "
                    f"B({x1_next:.3f},{x2_next:.3f},{q_next})={B_next:.6f}"
                )
            else:
                trajectory_info['steps'].append(
                    f"Step {step}: B={B_current:.6f} -> B'={B_next:.6f} (OK)"
                )
            
            x1, x2, q = x1_next, x2_next, q_next
        
        if not trajectory_info['condition1_ok'] or not trajectory_info['all_migrations_ok']:
            violations.append(trajectory_info)
    
    print(f"验证了 {n_trajectories} 条轨迹")
    print(f"发现 {len(violations)} 条违反轨迹")
    
    if violations:
        print("\n违反轨迹详情:")
        for viol in violations[:3]:  # 只显示前3个
            print(f"\n轨迹 {viol['id']}:")
            print(f"  初始状态: x1={viol['initial'][0]:.4f}, x2={viol['initial'][1]:.4f}, q={viol['initial'][2]}")
            if not viol['condition1_ok']:
                print(f"  ✗ 初始条件违反: B_init={viol.get('B_init', 'N/A'):.6f}")
            if not viol['all_migrations_ok']:
                print("  ✗ 迁移条件违反")
                for step_info in viol['steps'][-3:]:  # 显示最后3步
                    if "违反" in step_info:
                        print(f"    {step_info}")
    
    return violations

# 运行验证
if __name__ == "__main__":
    print("神经网络屏障证书数值验证")
    print("=" * 60)
    
    # 基本验证
    stats = numerical_verification(
        n_samples=100,  # 采样点数
        n_steps=1000,      # 每个点模拟步数
        seed=42          # 随机种子
    )
    
    print("\n" + "=" * 60)
    print("\n详细验证（从初始集合开始）:")
    
    # 详细验证
    violations = detailed_verification_from_initial(
        n_trajectories=100,
        max_steps=20
    )
    
    # 验证屏障函数的解析表达式
    print("\n" + "=" * 60)
    print("验证神经网络解析表达式:")
    
    # 创建神经网络
    barrier_net = create_barrier_network()
    
    # 手动计算屏障函数
    def B_analytical(x1, x2, q):
        """手动计算屏障函数的解析表达式"""
        # 网络参数
        W1 = barrier_net.fc1.weight.detach().numpy()
        b1 = barrier_net.fc1.bias.detach().numpy()
        W2 = barrier_net.fc2.weight.detach().numpy()
        b2 = barrier_net.fc2.bias.detach().numpy()
        
        # 输入向量
        inp = np.array([x1, x2, q])
        
        # 第1层: h = W1 @ inp + b1
        h = W1 @ inp + b1
        
        # 第2层: B = W2 @ h + b2
        B = W2 @ h + b2
        
        return float(B[0])
    
    # 测试几个点
    test_cases = [
        (0.1, 0.1, 1.0),
        (0.5, 0.5, 1.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.5, 1.0)
    ]
    
    print("\n神经网络计算 vs 解析表达式:")
    for x1, x2, q in test_cases:
        B_nn = barrier_net(torch.tensor(x1), torch.tensor(x2), torch.tensor(q)).item()
        B_analytic = B_analytical(x1, x2, q)
        
        diff = abs(B_nn - B_analytic)
        match = "✓" if diff < 1e-6 else "✗"
        
        print(f"  ({x1:.2f}, {x2:.2f}, {q}):")
        print(f"    NN: {B_nn:.6f}, Analytic: {B_analytic:.6f}, Diff: {diff:.6e} {match}")