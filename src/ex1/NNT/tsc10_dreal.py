import math
import torch
import torch.nn as nn
import torch.optim as optim
import dreal
import time
import json
import numpy as np
from datetime import datetime
import os

"""
一维 Kuramoto 振荡器系统
LTL 规约: G ¬Xu (永不进入不安全区域)
NBA 否定: F Xu (最终进入不安全区域)
- 阻断以q = 0的NBA接受迁移
- NBA 初始状态: q = 1 (安全状态)

验证目标: 迁移持久性证书 (Transition Persistence Certificate)
方法: 神经网络模板 + CEGIS
- 使用浅层神经网络 B(x, q) 作为屏障证书
- 通过梯度下降训练满足约束
- 使用 dReal SMT 求解器验证
- 反例引导的迭代精炼
"""
PI = 3.1415926
class BarrierNetwork(nn.Module):
    """用于屏障证书 B(x, q) 的浅层神经网络"""
    def __init__(self, input_dim=2, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU() # 添加 ReLU 激活函数
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, q):
        """
        输入:
            x: 状态变量 (tensor)
            q: 自动机状态 (tensor)
        返回:
            B(x, q): 屏障证书值
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)

        if x.dim() == 0:
            x = x.unsqueeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)

        inp = torch.cat([x.unsqueeze(-1), q.unsqueeze(-1)], dim=-1)
        h = self.fc1(inp)
        h = self.relu(h) # 应用 ReLU 激活
        out = self.fc2(h)
        return out.squeeze(-1)
    
    def save_model(self, filepath):
        """保存模型参数到文件"""
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存PyTorch模型
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"✓ 模型已保存到: {filepath}")
    
    def save_parameters_json(self, filepath):
        """保存网络参数为JSON格式"""
        params = {
            'W1': self.fc1.weight.detach().numpy().tolist(),
            'b1': self.fc1.bias.detach().numpy().tolist(),
            'W2': self.fc2.weight.detach().numpy().tolist(),
            'b2': self.fc2.bias.detach().numpy().tolist(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'architecture': f"{self.input_dim} -> {self.hidden_dim} -> 1",
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ 网络参数已保存为JSON: {filepath}")
    
    def save_parameters_txt(self, filepath):
        """保存网络参数为可读的文本格式"""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("屏障证书网络参数\n")
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"网络架构: {self.input_dim} -> {self.hidden_dim} -> 1\n")
            f.write(f"激活函数: ReLU\n\n")
            
            f.write("第一层权重 W1 (10x2):\n")
            W1 = self.fc1.weight.detach().numpy()
            for i in range(W1.shape[0]):
                f.write(f"  W1[{i}, :] = [{W1[i, 0]:.6f}, {W1[i, 1]:.6f}]\n")
            
            f.write("\n第一层偏置 b1 (10x1):\n")
            b1 = self.fc1.bias.detach().numpy()
            for i in range(b1.shape[0]):
                f.write(f"  b1[{i}] = {b1[i]:.6f}\n")
            
            f.write("\n第二层权重 W2 (1x10):\n")
            W2 = self.fc2.weight.detach().numpy()
            f.write("  W2[0, :] = [")
            for i in range(W2.shape[1]):
                f.write(f"{W2[0, i]:.6f}")
                if i < W2.shape[1] - 1:
                    f.write(", ")
            f.write("]\n")
            
            f.write(f"\n第二层偏置 b2 (1x1):\n")
            f.write(f"  b2 = {self.fc2.bias.detach().numpy()[0]:.6f}\n\n")
            
            f.write("解析表达式:\n")
            f.write("  B(x, q) = b2 + Σ_{i=1}^{10} W2[0,i] * ReLU(W1[i,0]*x + W1[i,1]*q + b1[i])\n\n")
            
            f.write("Python计算函数:\n")
            f.write("  def B(x, q):\n")
            f.write("      W1 = np.array(...)  # 从上面复制\n")
            f.write("      b1 = np.array(...)\n")
            f.write("      W2 = np.array(...)\n")
            f.write("      b2 = ...\n")
            f.write("      h_linear = W1[:,0]*x + W1[:,1]*q + b1\n")
            f.write("      h_relu = np.maximum(0, h_linear)\n")
            f.write("      return b2 + np.sum(W2[0,:] * h_relu)\n")
        
        print(f"✓ 网络参数已保存为文本: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """从文件加载模型"""
        checkpoint = torch.load(filepath)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✓ 模型已从 {filepath} 加载")
        return model


# ==================== 系统定义 ====================
def in_state_space(x_ce):
    """状态空间约束: x ∈ [0, 2π] (dreal格式)"""
    return dreal.And(x_ce >= 0, x_ce <= PI * 2)


def in_initial_set(x):
    """初始状态约束: x ∈ [0, π/9] (数值格式)"""
    return 4*PI/9 <= x <= PI*5 / 9


def in_initial_set_dreal(x_ce):
    """初始状态约束 (dreal格式)"""
    return dreal.And(x_ce >= 4*PI/9, x_ce <= PI*5 / 9)


def system_dynamics(x_ce):
    """系统动力学 (dreal格式)"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x_ce + Ts * Omega + Ts * K * dreal.sin(-x_ce) - 0.532 * x_ce ** 2 + 1.69


def system_dynamics_numeric(x):
    """系统动力学 (数值格式)"""
    Ts = 0.1
    Omega = 0.01
    K = 0.0006
    return x + Ts * Omega + Ts * K * math.sin(-x) - 0.532 * x ** 2 + 1.69


def in_unsafe_set(x_ce):
    """不安全状态约束: x ∈ [7π/9, 8π/9] (dreal格式)"""
    return dreal.And(x_ce >= PI / 9 * 7, x_ce <= PI / 9 * 8)


def in_unsafe_set_numeric(x):
    """不安全状态约束 (数值格式)"""
    return 7 * PI / 9 <= x <= 8 * PI / 9


def get_next_modes(q):
    """自动机状态转移"""
    if q == 1:
        return [0, 1]
    elif q == 0:
        return [0]
    else:
        raise ValueError(f"无效状态: {q}")


def compute_mode_transition(x, q):
    """计算模态转移"""
    if q == 1:
        if in_unsafe_set_numeric(x):
            return [0]
        else:
            return [1]
    else:
        return [0]


# ==================== 采样函数 ====================
def generate_samples(start, end, step):
    """生成采样点"""
    res = []
    for i in range(int(start * int(1/step)), int(end * int(1/step)) + 1):
        res.append(i * step)
    return res


def cartesian_product(set1, set2):
    """两个集合的笛卡尔积"""
    def connect(x1, x2):
        if not isinstance(x1, list) and not isinstance(x2, list):
            return [x1, x2]
        elif isinstance(x1, list) and not isinstance(x2, list):
            return x1 + [x2]
        elif not isinstance(x1, list) and isinstance(x2, list):
            return [x1] + x2
        else:
            return x1 + x2

    if not set1 or not set2:
        return []
    
    res = []
    for x1 in set1:
        for x2 in set2:
            res.append(connect(x1, x2))
    return res


def multi_cartesian_product(set1, *args):
    """多个集合的笛卡尔积"""
    res = cartesian_product(set1, args[0])
    for sp in args[1:]:
        res = cartesian_product(res, sp)
    return res


# ==================== 神经网络转dReal表达式 ====================
def convert_network_to_dreal(B_net):
    """
    将神经网络转换为dReal表达式
    B(x, q) = W2 @ (W1 @ [x, q] + b1) + b2
    """
    W1 = B_net.fc1.weight.detach().numpy()
    b1 = B_net.fc1.bias.detach().numpy()
    W2 = B_net.fc2.weight.detach().numpy()
    b2 = B_net.fc2.bias.detach().numpy()

    def barrier_dreal(x, q):
        """dReal表达式形式的屏障函数 (支持 ReLU)"""
        expr = b2[0]
        for i in range(len(b1)):
            # 线性部分
            h_linear = W1[i, 0] * x + W1[i, 1] * q + b1[i]
            # ReLU 激活: max(0, h_linear) -> 使用 if_then_else 替代
            h_relu = dreal.if_then_else(h_linear > 0, h_linear, 0)
            # 第二层加权求和
            expr += W2[0, i] * h_relu
        return expr

    return barrier_dreal


# ==================== 训练函数 ====================
def train_barrier_network(B_net, training_data, epochs=1000, lr=0.01):
    """
    训练屏障网络
    training_data: 包含 'init', 'unsafe', 'transition' 的字典
    """
    optimizer = optim.Adam(B_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0

        # 损失1: 初始状态 B(x0, q0) >= 0
        if training_data['init']:
            init_loss = 0.0
            for x0, q0 in training_data['init']:
                B_init = B_net(torch.tensor(x0, dtype=torch.float32), 
                              torch.tensor(q0, dtype=torch.float32))
                init_loss += torch.relu(-B_init + 0.01)  # 惩罚 B < 0
            loss_total += init_loss / len(training_data['init'])

        # 损失2: 不安全状态 B(xu, qu) < 0
        if training_data['unsafe']:
            unsafe_loss = 0.0
            for xu, qu in training_data['unsafe']:
                B_unsafe = B_net(torch.tensor(xu, dtype=torch.float32), 
                                torch.tensor(qu, dtype=torch.float32))
                unsafe_loss += torch.relu(B_unsafe + 0.01)  # 惩罚 B >= 0
            loss_total += unsafe_loss / len(training_data['unsafe'])

        # 损失3: 前向不变性 B(x,q) >= 0 => B(x',q') >= 0
        if training_data['transition']:
            trans_loss = 0.0
            for x, q, xp, qp in training_data['transition']:
                B_curr = B_net(torch.tensor(x, dtype=torch.float32), 
                              torch.tensor(q, dtype=torch.float32))
                B_next = B_net(torch.tensor(xp, dtype=torch.float32), 
                              torch.tensor(qp, dtype=torch.float32))
                trans_loss += torch.relu(-(B_next - B_curr))  # 惩罚 B_next < B_curr
            loss_total += trans_loss / len(training_data['transition'])

        loss_total.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")


# ==================== 验证函数 ====================
def verify_barrier_with_dreal(B_net):
    """
    使用dReal验证屏障证书
    返回: (是否验证通过, 反例字典)
    """
    # 创建dReal求解器
    solver = dreal.Context()
    solver.config.precision = 0.0001
    solver.SetLogic(dreal.Logic.QF_NRA)
    x_var = dreal.Variable('x')
    solver.DeclareVariable(x_var, 0, 2 * dreal.cos(0))

    barrier_dreal = convert_network_to_dreal(B_net)
    counterexamples = {'init': [], 'unsafe': [], 'transition': []}
    has_counterexample = False

    # 验证1: 初始状态条件
    solver.Push(2)
    solver.Assert(in_initial_set_dreal(x_var))
    solver.Assert(barrier_dreal(x_var, 1) < 0)
    model = solver.CheckSat()
    if model is not None:
        x_val = model[x_var].mid()
        print(f"  初始状态反例: x={x_val:.6f}, B(x,1)={barrier_dreal(x_val, 1)}")
        counterexamples['init'].append((float(x_val), 1))
        has_counterexample = True
    else:
        print("  ✓ 初始状态条件验证通过")
    solver.Pop(2)

    # 验证2: 不安全状态条件
    solver.Push(2)
    solver.Assert(in_state_space(x_var))
    solver.Assert(barrier_dreal(x_var, 0) >= 0)
    model = solver.CheckSat()
    if model is not None:
        x_val = model[x_var].mid()
        print(f"  不安全状态反例: x={x_val:.6f}, B(x,0)={barrier_dreal(x_val, 0)}")
        counterexamples['unsafe'].append((float(x_val), 0))
        has_counterexample = True
    else:
        print("  ✓ 不安全状态条件验证通过")
    solver.Pop(2)

    # 验证3: 模态q=1的转移条件
    has_transition_counterexample = False
    solver.Push(3)
    solver.Assert(in_state_space(x_var))
    solver.Assert(in_state_space(system_dynamics(x_var)))
    solver.Assert(barrier_dreal(x_var, 1) >= 0)
    
    for next_q in get_next_modes(1):
        solver.Push(2)
        if next_q == 0:
            solver.Assert(in_unsafe_set(x_var))
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)
        else:  # next_q == 1
            solver.Assert(dreal.Not(in_unsafe_set(x_var)))
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)
        
        model = solver.CheckSat()
        if model is not None:
            x_val = model[x_var].mid()
            xp_val = system_dynamics_numeric(x_val)
            print(f"  转移条件反例 (q=1->q={next_q}): x={x_val:.6f}, x'={xp_val:.6f}")
            counterexamples['transition'].append((float(x_val), 1, float(xp_val), next_q))
            has_counterexample = True
            has_transition_counterexample = True
        solver.Pop(2)
    solver.Pop(3)

    # 验证4: 模态q=0的转移条件
    solver.Push(3)
    solver.Assert(in_state_space(x_var))
    solver.Assert(in_state_space(system_dynamics(x_var)))
    solver.Assert(barrier_dreal(x_var, 0) >= 0)
    
    for next_q in get_next_modes(0):
        solver.Push(1)
        if next_q == 0:
            solver.Assert(barrier_dreal(system_dynamics(x_var), next_q) < 0)
        else:
            raise ValueError(f"无效转移: 0->{next_q}")
        
        model = solver.CheckSat()
        if model is not None:
            x_val = model[x_var].mid()
            xp_val = system_dynamics_numeric(x_val)
            print(f"  转移条件反例 (q=0->q={next_q}): x={x_val:.6f}, x'={xp_val:.6f}")
            counterexamples['transition'].append((float(x_val), 0, float(xp_val), next_q))
            has_counterexample = True
            has_transition_counterexample = True
        solver.Pop(1)
    solver.Pop(3)

    if not has_transition_counterexample:
        print("  ✓ 转移条件验证通过")

    return (not has_counterexample, counterexamples)


# ==================== 主CEGIS循环 ====================
def synthesize_barrier_certificate(save_dir="saved_models"):
    """主CEGIS循环合成屏障证书"""
    print("使用神经网络模板合成迁移安全证书")
    print("=" * 60)

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"barrier_net_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 初始化神经网络
    B_net = BarrierNetwork(input_dim=2, hidden_dim=10)

    # 生成采样数据
    x_samples = generate_samples(0, 2 * PI, 1)
    q_samples = [0, 1]
    
    print(f"\n采样点统计:")
    print(f"  x_samples: 状态空间采样点数量 = {len(x_samples)}")
    print(f"  q_samples: 模态空间大小 = {len(q_samples)}")
    
    all_states = multi_cartesian_product(x_samples, q_samples)
    print(f"  all_states: 完整状态空间大小 = {len(all_states)}")
    
    x0_samples = generate_samples(PI*4 / 9, PI*5 / 9, 0.1)
    print(f"  x0_samples: 初始状态采样点数量 = {len(x0_samples)}")
    
    initial_states = multi_cartesian_product(x0_samples, [1])
    print(f"  initial_states: 初始状态对数量 = {len(initial_states)}")
    
    unsafe_states = multi_cartesian_product(x_samples, [0])
    print(f"  unsafe_states: 不安全状态对数量 = {len(unsafe_states)}")
    
    # 计算转移样本数量
    transition_count = 0
    for x, q in all_states:
        qp_list = compute_mode_transition(x, q)
        transition_count += len(qp_list)
    print(f"  transition_samples: 转移样本数量 = {transition_count}")
    print(f"  模型保存目录: {model_dir}")
    print("-" * 40)

    # 初始化训练数据
    training_data = {
        'init': initial_states.copy(),
        'unsafe': unsafe_states.copy(),
        'transition': []
    }

    # 添加转移样本
    for x, q in all_states:
        xp = system_dynamics_numeric(x)
        qp_list = compute_mode_transition(x, q)
        for qp in qp_list:
            training_data['transition'].append((x, q, xp, qp))

    max_iterations = 20
    iteration = 0
    success = False

    # CEGIS循环
    while iteration < max_iterations:
        print(f"\n{'='*60}")
        print(f"CEGIS 迭代 #{iteration}")
        print(f"训练数据统计:")
        print(f"  init: 初始状态样本 = {len(training_data['init'])}")
        print(f"  unsafe: 不安全状态样本 = {len(training_data['unsafe'])}")
        print(f"  transition: 转移样本 = {len(training_data['transition'])}")

        # 步骤1: 训练候选证书
        print("\n步骤1: 训练神经网络...")
        train_barrier_network(B_net, training_data, epochs=1000, lr=0.01)

        # 步骤2: 使用dReal验证
        print("\n步骤2: 使用dReal验证...")
        verified, counterexamples = verify_barrier_with_dreal(B_net)

        if verified:
            print(f"\n{'='*60}")
            print("✓ 验证通过! 成功合成屏障证书")
            print("=" * 60)
            success = True
            break
        else:
            # 步骤3: 添加反例到训练数据
            print("\n步骤3: 添加反例到训练数据...")
            added_count = 0
            for ce in counterexamples['init']:
                training_data['init'].append(ce)
                added_count += 1
            for ce in counterexamples['unsafe']:
                training_data['unsafe'].append(ce)
                added_count += 1
            for ce in counterexamples['transition']:
                training_data['transition'].append(ce)
                added_count += 1
            print(f"  添加了 {added_count} 个反例到训练数据")
            print(f"  新增训练数据统计:")
            print(f"    初始状态: {len(counterexamples['init'])} 个")
            print(f"    不安全状态: {len(counterexamples['unsafe'])} 个")
            print(f"    转移样本: {len(counterexamples['transition'])} 个")

        iteration += 1

    if iteration >= max_iterations:
        print(f"\n{'='*60}")
        print("✗ 超过最大迭代次数")
        print("=" * 60)
    elif not success:
        print(f"\n{'='*60}")
        print("✗ 无法合成屏障证书")
        print("=" * 60)

    return B_net, success, model_dir


# ==================== 主程序 ====================
if __name__ == "__main__":
    start_time = time.time()
    barrier_net, success, model_dir = synthesize_barrier_certificate()
    end_time = time.time()

    if success:
        print(f"\n总用时: {end_time - start_time:.4f} 秒")
        print("\n最终屏障证书网络参数:")
        print("-" * 50)
        print("架构: 2 -> 10 -> 1 (ReLU 激活网络)")
        print("\n第一层权重 (W1):")
        print(barrier_net.fc1.weight.detach().numpy())
        print("\n第一层偏置 (b1):")
        print(barrier_net.fc1.bias.detach().numpy())
        print("\n第二层权重 (W2):")
        print(barrier_net.fc2.weight.detach().numpy())
        print("\n第二层偏置 (b2):")
        print(barrier_net.fc2.bias.detach().numpy())
        print("-" * 50)
        print(f"\n解析表达式: B(x, q) = W2 @ ReLU(W1 @ [x, q] + b1) + b2")
        
        # 保存模型
        print("\n" + "=" * 60)
        print("保存模型参数...")
        print("=" * 60)
        
        # 保存PyTorch模型
        model_path = os.path.join(model_dir, "barrier_net.pth")
        barrier_net.save_model(model_path)
        
