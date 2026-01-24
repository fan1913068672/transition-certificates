import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dreal import *
import random
import time

class CEGISTemplateNetwork(nn.Module):
    """
    完全匹配CEGIS模板的神经网络
    """
    def __init__(self):
        super(CEGISTemplateNetwork, self).__init__()
        
        # 10个系数，对应模板中的c[0]到c[9]
        self.coeff = nn.Parameter(torch.zeros(10))
        
        # 初始化接近目标值
        with torch.no_grad():
            # 根据您的成功结果初始化
            # B(x1,x2) = 27*I_VF - 1*x1*I_VF
            # 对应: c[2]=27, c[5]=-1, 其他为0
            self.coeff[2] = 27.0  # I_VF系数
            self.coeff[5] = -1.0  # x1*I_VF系数
        
        self.epsilon = 0.25  # 使用CEGIS的epsilon
    
    def forward(self, x1, x2):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
            
        if x1.dim() == 0:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 0:
            x2 = x2.unsqueeze(0)
        
        # 计算条件
        in_x0 = ((x1 >= 21) & (x1 <= 24) & (x2 >= 21) & (x2 <= 24)).float()
        in_vf = ((x1 >= 20) & (x1 <= 26) & (x2 >= 20) & (x2 <= 26)).float()
        x1_gt_x2 = (x1 > x2).float()
        
        # 计算max(x1, x2)
        max_val = x1_gt_x2 * x1 + (1 - x1_gt_x2) * x2
        
        # 计算模板的各项
        terms = torch.stack([
            torch.ones_like(x1),            # c[0] * 1
            in_x0,                          # c[1] * I_X0
            in_vf,                          # c[2] * I_VF
            x1 * in_x0,                     # c[3] * x1 * I_X0
            x2 * in_x0,                     # c[4] * x2 * I_X0
            x1 * in_vf,                     # c[5] * x1 * I_VF
            x2 * in_vf,                     # c[6] * x2 * I_VF
            max_val,                        # c[7] * max(x1, x2)
            x1**2,                          # c[8] * x1**2
            x2**2                           # c[9] * x2**2
        ], dim=0)  # shape: (10, batch)
        
        # 加权求和
        B = torch.sum(self.coeff.unsqueeze(1) * terms, dim=0)
        
        return B
    
    def get_epsilon(self):
        return self.epsilon
    
    def get_coeff(self):
        return self.coeff.detach().numpy()
    
    def get_expression(self):
        """生成CEGIS格式的表达式"""
        coeff = self.get_coeff()
        
        # 构建各项
        terms = [
            (0, "", coeff[0]),
            (1, "I_X0", coeff[1]),
            (2, "I_VF", coeff[2]),
            (3, "x1·I_X0", coeff[3]),
            (4, "x2·I_X0", coeff[4]),
            (5, "x1·I_VF", coeff[5]),
            (6, "x2·I_VF", coeff[6]),
            (7, "max(x1,x2)", coeff[7]),
            (8, "x1²", coeff[8]),
            (9, "x2²", coeff[9])
        ]
        
        # 筛选非零项
        non_zero_terms = []
        for idx, term_str, coeff_val in terms:
            if abs(coeff_val) > 1e-6:
                if idx == 0:
                    non_zero_terms.append(f"{coeff_val:.6f}")
                else:
                    sign = "+" if coeff_val >= 0 else "-"
                    non_zero_terms.append(f"{sign} {abs(coeff_val):.6f}·{term_str}")
        
        if not non_zero_terms:
            expr = "0"
        else:
            expr = " ".join(non_zero_terms)
            if expr.startswith("+"):
                expr = expr[2:]  # 去掉开头的 "+ "
        
        return f"B(x1,x2) = {expr}"

def get_Bp_dreal_cegis(B_net):
    """为CEGIS模板网络生成dReal表达式"""
    coeff = B_net.get_coeff()
    
    def Bp_c(x1, x2):
        # 计算条件
        in_x0_cond = logical_and(logical_and(x1 >= 21, x1 <= 24),
                                logical_and(x2 >= 21, x2 <= 24))
        in_vf_cond = logical_and(logical_and(x1 >= 20, x1 <= 26),
                                logical_and(x2 >= 20, x2 <= 26))
        x1_gt_x2_cond = x1 > x2
        
        # 计算max(x1, x2)
        max_val = if_then_else(x1_gt_x2_cond, x1, x2)
        
        # 构建模板表达式
        expr = (float(coeff[0]) + 
                float(coeff[1]) * if_then_else(in_x0_cond, 1.0, 0.0) + 
                float(coeff[2]) * if_then_else(in_vf_cond, 1.0, 0.0) + 
                float(coeff[3]) * x1 * if_then_else(in_x0_cond, 1.0, 0.0) + 
                float(coeff[4]) * x2 * if_then_else(in_x0_cond, 1.0, 0.0) + 
                float(coeff[5]) * x1 * if_then_else(in_vf_cond, 1.0, 0.0) + 
                float(coeff[6]) * x2 * if_then_else(in_vf_cond, 1.0, 0.0) + 
                float(coeff[7]) * max_val + 
                float(coeff[8]) * x1**2 + 
                float(coeff[9]) * x2**2)
        
        return expr
    
    return Bp_c

def train_cegis_network(B_net, training_data, epochs=30, lr=0.1):
    """训练CEGIS模板网络"""
    optimizer = optim.Adam([B_net.coeff], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0.0
        epsilon = B_net.get_epsilon()
        
        
        
        # 2. 非递增损失
        if len(training_data['non_inc']) > 0:
            non_inc_loss = 0.0
            for x1, x2, x1p, x2p in training_data['non_inc']:
                B_curr = B_net(x1, x2)
                B_next = B_net(x1p, x2p)
                violation = torch.relu(B_next - B_curr)
                non_inc_loss += violation
            loss_total += 10.0 * non_inc_loss / len(training_data['non_inc'])
        
        # 3. 严格递减损失
        if len(training_data['strict_dec']) > 0:
            strict_dec_loss = 0.0
            for x1, x2, x1p, x2p in training_data['strict_dec']:
                B_curr = B_net(x1, x2)
                B_next = B_net(x1p, x2p)
                violation = torch.relu(B_next - B_curr + epsilon)
                strict_dec_loss += violation
            loss_total += 100.0 * strict_dec_loss / len(training_data['strict_dec'])
        
        loss_total.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            coeff = B_net.get_coeff()
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")
            # 打印主要系数
            if abs(coeff[2]) > 1e-3 or abs(coeff[5]) > 1e-3:
                print(f"    主要系数: c2(I_VF)={coeff[2]:.3f}, c5(x1·I_VF)={coeff[5]:.3f}")

def synthesize_cegis_template():
    """使用CEGIS模板网络合成Barrier函数"""
    print("=" * 70)
    print("CEGIS模板神经网络合成")
    print("模板结构: 10个系数，完全匹配CEGIS")
    print("=" * 70)
    
    B_net = CEGISTemplateNetwork()
    
    # 生成训练数据
    X1_Samples = step_sample(20, 34, 1.0)
    X2_Samples = step_sample(20, 34, 1.0)
    All_Samples = state_space_product(X1_Samples, X2_Samples)
    
    # 训练数据
    training_data = {'non_inc': [], 'strict_dec': []}
    

    
    # 转移数据
    for x1, x2 in All_Samples:
        x1p, x2p = f_m(x1, x2)
        training_data['non_inc'].append((x1, x2, x1p, x2p))
        if In_VF(x1, x2):
            training_data['strict_dec'].append((x1, x2, x1p, x2p))
    
    MAX_ITER = 100
    for iter in range(MAX_ITER):
        print(f"\n迭代 {iter+1}/{MAX_ITER}")
        
        # 训练
        train_cegis_network(B_net, training_data, epochs=200, lr=0.05)
        
        # 验证
        config = Config()
        config.precision = 1e-4
        
        x1_ce = Variable('x1_ce')
        x2_ce = Variable('x2_ce')
        x1p_ce, x2p_ce = f_cond(x1_ce, x2_ce)
        Bp_c = get_Bp_dreal_cegis(B_net)
        epsilon_val = B_net.get_epsilon()
        
        verified = True
        
        # 检查非递增
        print("  验证非递增...")
        formula = logical_and(
            logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
            logical_and(logical_not(In_VF_Cond(x1_ce, x2_ce)),
                       Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce))
        )
        result = CheckSatisfiability(formula, config)
        if result:
            box = result
            x1_val = box[x1_ce].mid()
            x2_val = box[x2_ce].mid()
            training_data['non_inc'].append((x1_val, x2_val, f_m(x1_val, x2_val)[0], f_m(x1_val, x2_val)[1]))
            verified = False
            print(f"    找到反例: ({x1_val:.3f}, {x2_val:.3f})")
        else:
            print("    ✓ 通过")
        
        # 检查严格递减
        print(f"  验证严格递减 (ε={epsilon_val:.6f})...")
        formula = logical_and(
            logical_and(In_X_Cond(x1_ce, x2_ce), In_X_Cond(x1p_ce, x2p_ce)),
            logical_and(In_VF_Cond(x1_ce, x2_ce),
                       Bp_c(x1_ce, x2_ce) < Bp_c(x1p_ce, x2p_ce) + epsilon_val)
        )
        result = CheckSatisfiability(formula, config)
        if result:
            box = result
            x1_val = box[x1_ce].mid()
            x2_val = box[x2_ce].mid()
            training_data['strict_dec'].append((x1_val, x2_val, f_m(x1_val, x2_val)[0], f_m(x1_val, x2_val)[1]))
            verified = False
            print(f"    找到反例: ({x1_val:.3f}, {x2_val:.3f})")
        else:
            print("    ✓ 通过")
        
        if verified:
            print("\n" + "="*70)
            print("✓ 合成成功!")
            print("="*70)
            return B_net, True
    
    return B_net, False

# 辅助函数
alpha = 0.004
theta = 0.01
Te = 0
Th = 40
mu = 0.15

def u(x_curr):
    return 0.59 - 0.011 * x_curr

def f_cond(x1, x2):
    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te
    return x1_next, x2_next

def f_m(x1, x2):
    x1_next = (1 - 2 * alpha - theta - mu * u(x1)) * x1 + x2 * alpha + mu * Th * u(x1) + theta * Te
    x2_next = x1 * alpha + (1 - 2 * alpha - theta - mu * u(x2)) * x2 + mu * Th * u(x2) + theta * Te
    return x1_next, x2_next

def In_VF(x1, x2):
    return x1 >= 20 and x1 <= 26 and x2 >= 20 and x2 <= 26

def In_VF_Cond(x1_ce, x2_ce):
    return logical_and(logical_and(x1_ce >= 20, x1_ce <= 26),
                      logical_and(x2_ce >= 20, x2_ce <= 26))

def In_X_Cond(x1_ce, x2_ce):
    return logical_and(logical_and(x1_ce >= 20, x1_ce <= 34),
                      logical_and(x2_ce >= 20, x2_ce <= 34))

def step_sample(a, b, s):
    res = []
    for i in range(a * int(1 / s), b * int(1 / s) + 1):
        res.append(i * s)
    return res

def space_product(s1, s2):
    def conn(x1, x2):
        if type(x1) != list and type(x2) != list:
            return [x1, x2]
        elif type(x1) == list and type(x2) != list:
            return x1 + [x2]
        elif type(x1) != list and type(x2) == list:
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
    res = space_product(s1, args[0])
    for sp in args[1:]:
        res = space_product(res, sp)
    return res

# 主程序
if __name__ == "__main__":
    start_time = time.time()
    B_net, success = synthesize_cegis_template()
    end_time = time.time()
    
    if success:
        coeff = B_net.get_coeff()
        expr = B_net.get_expression()
        
        print("\n合成结果:")
        print("-" * 70)
        print(expr)
        print(f"ε = {B_net.get_epsilon()}")
        print("-" * 70)
        
        print("\n详细系数:")
        for i, c in enumerate(coeff):
            if abs(c) > 1e-6:
                if i == 0:
                    print(f"  c[{i}] (常数项) = {c:.6f}")
                elif i == 1:
                    print(f"  c[{i}] (I_X0) = {c:.6f}")
                elif i == 2:
                    print(f"  c[{i}] (I_VF) = {c:.6f}")
                elif i == 3:
                    print(f"  c[{i}] (x1·I_X0) = {c:.6f}")
                elif i == 4:
                    print(f"  c[{i}] (x2·I_X0) = {c:.6f}")
                elif i == 5:
                    print(f"  c[{i}] (x1·I_VF) = {c:.6f}")
                elif i == 6:
                    print(f"  c[{i}] (x2·I_VF) = {c:.6f}")
                elif i == 7:
                    print(f"  c[{i}] (max(x1,x2)) = {c:.6f}")
                elif i == 8:
                    print(f"  c[{i}] (x1²) = {c:.6f}")
                elif i == 9:
                    print(f"  c[{i}] (x2²) = {c:.6f}")
        
        # 比较目标函数
        print("\n验证点对比 (目标: 27*I_VF - x1*I_VF):")
        test_points = [(21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]
        for x1, x2 in test_points:
            B_val = B_net(x1, x2).item()
            B_target = 27 * In_VF(x1, x2) - x1 * In_VF(x1, x2)
            in_vf = In_VF(x1, x2)
            print(f"  ({x1},{x2}) VF={in_vf}: 网络={B_val:.3f}, 目标={B_target:.3f}, 差值={abs(B_val-B_target):.3f}")
    
    print(f"\n总耗时: {end_time - start_time:.2f}秒")