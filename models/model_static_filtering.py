import torch
import torch.nn as nn
from utils.layers import filter_static_agents

class StaticFilteringModel(nn.Module):
    """简化的静态代理过滤模型，仅保留过滤逻辑。"""
    def __init__(self, feature_dim, time_steps, head_num=4):
        super(StaticFilteringModel, self).__init__()
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.head_num = head_num
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=head_num)

    def forward(self, features, positions, threshold):
        """
        使用静态代理过滤处理输入。
        Args:
            features: 输入特征 [A, T, D]
            positions: 代理位置 [A, T, 2]
            threshold: 静态过滤阈值
        """
        A, T, D = features.shape

        # 过滤静态代理
        active_agent_mask = filter_static_agents(positions, threshold)
        active_agents = torch.nonzero(active_agent_mask).squeeze()

        filtered_features = features[active_agents]
        q = filtered_features.permute(1, 0, 2)  # [T, A_filtered, D]
        k, v = q.clone(), q.clone()

        attn_output, _ = self.attention(q, k, v)
        return attn_output.permute(1, 0, 2)  # [A_filtered, T, D]
