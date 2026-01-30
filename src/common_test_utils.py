import torch
import torch.nn as nn
import os
import glob
from pathlib import Path

class BarrierNetwork(nn.Module):
    """通用屏障证书神经网络结构"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, *args):
        # 处理不定数量的状态输入
        if len(args) < 1:
            raise ValueError("At least one input (state/mode) is required")
        
        tensors = []
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                t = torch.tensor(arg, dtype=torch.float32)
            else:
                t = arg.clone().detach().to(torch.float32)
            
            if t.dim() == 0:
                t = t.unsqueeze(0)
            tensors.append(t.unsqueeze(-1))
            
        inp = torch.cat(tensors, dim=-1)
        h = self.fc1(inp)
        h = self.relu(h)
        out = self.fc2(h)
        return out.squeeze(-1)

def find_latest_model(base_path):
    """自动寻找最新的 .pth 模型文件"""
    search_path = os.path.join(base_path, "saved_models", "**", "*.pth")
    files = glob.glob(search_path, recursive=True)
    if not files:
        # 尝试直接在 saved_models 下寻找
        search_path = os.path.join(base_path, "saved_models", "*.pth")
        files = glob.glob(search_path)
        
    if not files:
        raise FileNotFoundError(f"No .pth files found in {base_path}/saved_models")
    
    # 按修改时间排序
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_model(model_path, input_dim, hidden_dim=None):
    """加载模型参数"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 自动获取 hidden_dim
    if hidden_dim is None:
        if 'hidden_dim' in checkpoint:
            hidden_dim = checkpoint['hidden_dim']
        else:
            # 从 state_dict 推断
            hidden_dim = checkpoint['state_dict']['fc1.bias'].shape[0]
            
    model = BarrierNetwork(input_dim, hidden_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
