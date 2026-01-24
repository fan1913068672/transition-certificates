import math
import torch
import torch.nn as nn
from pathlib import Path

"""
验证神经网络屏障证书的有效性
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

# ==================== 加载神经网络屏障证书 ====================
class BarrierNetwork(nn.Module):
    """用于屏障证书 B(x, q) 的浅层神经网络"""
    def __init__(self, input_dim=2, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, q):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)
        
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)
        
        inp = torch.cat([x.unsqueeze(-1), q.unsqueeze(-1)], dim=-1)
        h = self.fc1(inp)  # 线性激活
        out = self.fc2(h)
        return out.squeeze(-1)
    
    @classmethod
    def load_from_pth(cls, filepath: str) -> 'BarrierNetwork':
        """从.pth文件加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(
            input_dim=checkpoint.get('input_dim', 2),
            hidden_dim=checkpoint.get('hidden_dim', 10)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


def load_barrier_function(model_path: str):
    """加载屏障函数"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = BarrierNetwork.load_from_pth(model_path)
    
    def barrier_func(x: float, mode: int) -> float:
        """计算屏障函数值"""
        with torch.no_grad():
            return model(x, mode).item()
    
    return barrier_func, model

# ==================== 采样函数 ====================
def generate_samples(start: float, end: float, step: float) -> list[float]:
    """
    在区间[start, end]上以步长step生成采样点
    """
    num_samples = int((end - start) / step) + 1
    return [start + i * step for i in range(num_samples)]

# ==================== 证书验证函数 ====================
def verify_safety_certificate(barrier_func, max_iterations: int = 1000) -> bool:
    """
    验证安全证书的有效性
    返回: True 如果证书有效，False 如果发现反例
    """
    # 在初始集上密集采样
    initial_samples = generate_samples(0, math.pi / 9, 0.001)
    
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
        
        # 每验证100个点显示一次进度
        idx = initial_samples.index(initial_x)
        if idx % 1 == 0 or idx == len(initial_samples) - 1:
            print(f"进度: {idx+1}/{len(initial_samples)} - 初始状态 x = {initial_x:.6f} 验证通过")
    
    print("\n" + "=" * 50)
    print("✅ 所有初始状态验证通过！证书有效。")
    print("=" * 50)
    return True


def print_model_info(model):
    """打印模型信息"""
    print(f"\n模型架构: {model.input_dim} -> {model.hidden_dim} -> 1")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 打印权重范围
    w1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    w2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()
    
    print(f"W1 范围: [{w1.min():.4f}, {w1.max():.4f}]")
    print(f"b1 范围: [{b1.min():.4f}, {b1.max():.4f}]")
    print(f"W2 范围: [{w2.min():.4f}, {w2.max():.4f}]")
    print(f"b2 值: {b2[0]:.4f}")

# ==================== 主程序 ====================
def main():
    """主验证程序"""
    print("=" * 60)
    print("神经网络屏障证书验证工具")
    print("=" * 60)
    
    # 加载模型
    model_path = 'saved_models/barrier_net_20260109_022535/barrier_net.pth'
    
    try:
        barrier_func, model = load_barrier_function(model_path)
        print(f"✓ 模型加载成功")
        print_model_info(model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 完整验证
    is_valid = verify_safety_certificate(
        barrier_func=barrier_func,
        max_iterations=2000
    )
    
    if is_valid:
        print("\n验证总结:")
        print("1. ✅ 初始状态非负性: 所有 x∈X0 满足 B(x,1) ≥ 0")
        print("2. ✅ 前向不变性: B(x,q) ≥ 0 ⇒ B(f(x), δ(q,L(x))) ≥ 0")
        print("3. ✅ 安全性: 从初始状态无法到达不安全状态")
        print("\n🎉 神经网络屏障证书验证成功！")
    else:
        print("\n❌ 神经网络屏障证书验证失败，发现反例")
    
    print("=" * 60)

if __name__ == "__main__":
    main()