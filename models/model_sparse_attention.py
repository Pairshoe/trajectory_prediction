import torch
import torch.nn as nn
from utils.layers import generate_sparse_mask

class SparseAttentionModel(nn.Module):
    """简化的稀疏注意力模型，仅保留注意力逻辑。"""
    def __init__(self, feature_dim, time_steps, head_num=4):
        super(SparseAttentionModel, self).__init__()
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.head_num = head_num
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=head_num)

    def forward(self, features, positions, max_distance, time_window):
        """
        使用稀疏注意力处理输入。
        Args:
            features: 输入特征 [A, T, D]
            positions: 代理位置 [A, T, 2]
            max_distance: 最大代理距离
            time_window: 时间窗口大小
        """
        A, T, D = features.shape
        sparse_mask = generate_sparse_mask(positions, max_distance, time_window)

        q = features.permute(1, 0, 2)  # [T, A, D]
        k, v = q.clone(), q.clone()

        # 应用稀疏掩码
        attn_output, _ = self.attention(q, k, v, attn_mask=sparse_mask)
        return attn_output.permute(1, 0, 2)  # [A, T, D]
