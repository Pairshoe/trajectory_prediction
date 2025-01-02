import torch
import torch.nn as nn
from utils import SelfAttLayer_dec, Permute4Batchnorm

class Decoder(nn.Module):
    """解码器模块。"""
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4, num_features=6):
        super().__init__()
        self.layer_T = nn.Sequential(
            nn.Linear(feature_dim + num_features, feature_dim),
            nn.ReLU()
        )
        self.layer_U = SelfAttLayer_dec(time_steps, feature_dim, head_num, k, across_time=True)
        self.layer_Z2 = nn.Linear(6, 6)

    def forward(self, encodings, batch_mask, padding_mask, hidden_mask=None):
        x = encodings.unsqueeze(0).repeat(6, 1, 1, 1)
        x = self.layer_T(x)
        x = self.layer_U(x, batch_mask=batch_mask, padding_mask=padding_mask)
        x = self.layer_Z2(x)
        return x
