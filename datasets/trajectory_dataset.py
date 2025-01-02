import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """轨迹数据集。"""
    def __init__(self, num_samples=100, seq_len=91):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # 生成轨迹数据
        self.data = []
        for _ in range(num_samples):
            # 生成主曲线轨迹
            t = np.linspace(0, 1, seq_len)
            
            # 添加一些随机曲率
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            radius = np.random.uniform(30, 100)
            
            # 基础轨迹
            x = t * 100
            y = -t * 100
            
            # 添加曲线
            x += radius * np.sin(t * 2 * np.pi + angle)
            y += radius * np.cos(t * 2 * np.pi + angle)
            
            # 添加一些噪声
            x += np.random.normal(0, 2, seq_len)
            y += np.random.normal(0, 2, seq_len)
            
            # 组合成轨迹
            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        trajectory = self.data[idx]
        # 转换为张量并添加掩码
        x = torch.FloatTensor(trajectory)
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        mask[:10] = False  # 掩盖前10个时间步
        
        return x, mask