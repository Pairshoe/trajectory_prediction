import torch
import torch.nn as nn
from utils import View, Permute4Batchnorm, SelfAttLayer, CrossAttLayer

class Encoder(nn.Module):
    """编码器模块。"""
    def __init__(self, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, time_steps=91, 
                 feature_dim=256, head_num=4, max_dynamic_rg=16, max_static_rg=1400, k=4):
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.head_dim = feature_dim // head_num

        # 代理特征
        self.layer_A = nn.Sequential(
            nn.Linear(in_feat_dim, feature_dim),
            nn.ReLU(),
            Permute4Batchnorm((0, 2, 1)),
            nn.BatchNorm1d(feature_dim),
            Permute4Batchnorm((0, 2, 1))
        )

        # 自注意力层
        self.layer_D = SelfAttLayer(time_steps, feature_dim, head_num, k, across_time=True)
        self.layer_E = SelfAttLayer(time_steps, feature_dim, head_num, k, across_time=False)
        self.layer_J = CrossAttLayer(time_steps, feature_dim, head_num, k)

    def forward(self, state_feat, agent_batch_mask, padding_mask, hidden_mask, 
                road_feat, roadgraph_valid, traffic_light_feat, traffic_light_valid,
                agent_rg_mask, agent_traffic_mask):
        state_feat[hidden_mask == False] = -1
        A_ = self.layer_A(state_feat)
        output, _, _, _ = self.layer_D(A_, agent_batch_mask, padding_mask, hidden_mask)
        output, _, _, _ = self.layer_E(output, agent_batch_mask, padding_mask, hidden_mask)
        output, _, _, _ = self.layer_J(output, road_feat, agent_rg_mask, padding_mask, roadgraph_valid)
        return output
