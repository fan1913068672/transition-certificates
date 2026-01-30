import math
import torch
import random
import os
import sys
from pathlib import Path

# 添加上级目录到路径以便导入 common_test_utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_test_utils import load_model, find_latest_model

# ==================== 系统参数 ====================
SAMPLING_TIME = 0.1
NATURAL_FREQUENCY = 0.01
COUPLING_COEFFICIENT = 0.0006
QUADRATIC_TERM = 0.532
CONSTANT_TERM = 1.69

def in_initial_set(x):
    return 0 <= x <= math.pi / 9

def in_unsafe_set(x):
    return 7 * math.pi / 9 <= x <= 8 * math.pi / 9

def system_dynamics(x):
    return (x + SAMPLING_TIME * NATURAL_FREQUENCY + 
            SAMPLING_TIME * COUPLING_COEFFICIENT * math.sin(-x) - 
            QUADRATIC_TERM * x ** 2 + CONSTANT_TERM)

def mode_transition(q, x):
    if q == 1:
        return 0 if in_unsafe_set(x) else 1
    return 0

# ==================== 测试逻辑 ====================
def run_test():
    print("="*70)
    print("Testing Neural Barrier Certificate for Ex1 (1D)")
    print("="*70)

    # 动态加载模型
    try:
        model_path = find_latest_model(Path(__file__).parent)
        print(f"✓ Loading latest model: {model_path}")
        model = load_model(model_path, input_dim=2)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    num_trajectories = 100
    max_steps = 1000
    passed_count = 0
    
    print(f"Running {num_trajectories} trajectories, {max_steps} steps each...")

    for i in range(num_trajectories):
        # 采样初始点
        x = random.uniform(0, math.pi / 9)
        q = 1
        
        # 初始条件检查
        B_init = model(x, q).item()
        if B_init < -1e-6:
            print(f"Trajectory {i+1}: ❌ Initial Condition Violated! B={B_init:.6f}")
            continue

        violation = False
        for step in range(max_steps):
            x_next = system_dynamics(x)
            q_next = mode_transition(q, x)
            
            B_curr = model(x, q).item()
            B_next = model(x_next, q_next).item()
            
            # 检查安全性: q=0 时 B 必须 < 0 (或者说 B>=0 时不能进入 q=0)
            if q_next == 0 and B_curr >= -1e-6:
                print(f"Trajectory {i+1}: ❌ Safety Violated at step {step}! Entered q=0 from B={B_curr:.6f}")
                violation = True
                break
                
            # 检查不变性: B_curr >= 0 => B_next >= 0
            if B_curr >= -1e-6 and B_next < -1e-6:
                print(f"Trajectory {i+1}: ❌ Invariance Violated at step {step}! B_curr={B_curr:.6f}, B_next={B_next:.6f}")
                violation = True
                break
            
            x, q = x_next, q_next
            
        if not violation:
            passed_count += 1
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{num_trajectories} trajectories completed.")

    print("\n" + "="*70)
    print("Test Summary")
    print("-" * 70)
    print(f"Total Trajectories: {num_trajectories}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {num_trajectories - passed_count}")
    
    if passed_count == num_trajectories:
        print("\n🎉 All tests passed! The barrier certificate is valid.")
    else:
        print("\n❌ Some tests failed. Please check the model or training process.")
    print("="*70)

if __name__ == "__main__":
    run_test()
