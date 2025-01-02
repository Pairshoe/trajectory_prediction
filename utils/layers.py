import torch
import torch.nn as nn

class View(nn.Module):
    """用于调整张量形状的模块。"""
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Permute4Batchnorm(nn.Module):
    """在 BatchNorm 前调整张量的维度顺序。"""
    def __init__(self, order):
        super(Permute4Batchnorm, self).__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)

class ScaleLayer(nn.Module):
    """用于缩放张量的层。"""
    def __init__(self, shape, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

    def forward(self, input):
        return input * self.scale

class SelfAttLayer(nn.Module):
    """自注意力层。"""
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()

        self.viewmodule_ = View((-1, time_steps, head_num, feature_dim // head_num))
        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_K_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_V_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_Q0_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_Q_ = ScaleLayer(feature_dim // head_num)

        self.scale = torch.sqrt(torch.FloatTensor([head_num]))

        self.layer_Y2_ = nn.Sequential(
            View((-1, time_steps, feature_dim)),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        self.layer_F1_ = nn.Sequential(
            nn.Linear(feature_dim, k * feature_dim),
            nn.ReLU()
        )
        self.layer_F2_ = nn.Sequential(
            nn.Linear(k * feature_dim, feature_dim),
            nn.ReLU()
        )
        self.layer_Z_ = nn.LayerNorm(feature_dim)

        self.across_time = across_time

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        device = x.device
        x = self.layer_X_(x)
        K = self.layer_K_(x)
        V = self.layer_V_(x)
        Q0 = self.layer_Q0_(x)
        Q = self.layer_Q_(Q0)  # Q,K,V -> [A,T,H,d]

        scale = self.scale.to(device)

        if self.across_time:
            Q, K, V = Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3)  # [A,H,T,d]
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale  # [A,H,T,T]
            if padding_mask is not None:
                energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == False, -1e10)
            attention = torch.softmax(energy, dim=-1)  # [A,H,T,T]
            Y1_ = torch.matmul(attention, V)  # [A,H,T,d]
            Y1_ = Y1_.permute(0, 2, 1, 3).contiguous()  # [A,T,H,d]
        else:
            Q, K, V = Q.permute(1, 2, 0, 3), K.permute(1, 2, 0, 3), V.permute(1, 2, 0, 3)  # [T,H,A,d]
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale  # [T,H,A,A]
            if batch_mask is not None:
                energy = energy.masked_fill(batch_mask.unsqueeze(1).unsqueeze(0) == 0, -1e10)
            if padding_mask is not None:
                energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == False, -1e10)
            attention = torch.softmax(energy, dim=-1)  # [T,H,A,A]
            Y1_ = torch.matmul(attention, V)  # [T,H,A,d]
            Y1_ = Y1_.permute(2, 0, 1, 3).contiguous()  # [A,T,H,d]

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + x
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V  # [A,T,D], [A,H,T,d]*3

class CrossAttLayer(nn.Module):
    """交叉注意力层。"""
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4):
        super().__init__()

        self.viewmodule_ = View((-1, time_steps, head_num, feature_dim // head_num))
        self.layer_X_ = nn.LayerNorm(feature_dim)
        
        self.layer_K_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_V_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_Q0_ = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            self.viewmodule_
        )
        self.layer_Q_ = ScaleLayer(feature_dim // head_num)

        self.scale = torch.sqrt(torch.FloatTensor([head_num]))

        self.layer_Y2_ = nn.Sequential(
            View((-1, time_steps, feature_dim)),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        self.layer_F1_ = nn.Sequential(
            nn.Linear(feature_dim, k * feature_dim),
            nn.ReLU()
        )
        self.layer_F2_ = nn.Sequential(
            nn.Linear(k * feature_dim, feature_dim),
            nn.ReLU()
        )
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, agent, rg, agent_rg_mask, padding_mask, rg_valid_mask):
        device = agent.device
        agent = self.layer_X_(agent)
        rg = self.layer_X_(rg)
        
        K = self.layer_K_(rg)  # [G,T,H,d]
        V = self.layer_V_(rg)  # [G,T,H,d]
        Q0 = self.layer_Q0_(agent)
        Q = self.layer_Q_(Q0)  # [A,T,H,d]

        scale = self.scale.to(device)

        Q, K, V = Q.permute(1, 2, 0, 3), K.permute(1, 2, 0, 3), V.permute(1, 2, 0, 3)  # Q -> [T,H,A,d] / K,V -> [T,H,G,d]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale  # [T,H,A,G]

        if agent_rg_mask is not None:
            energy = energy.masked_fill(agent_rg_mask.unsqueeze(1).unsqueeze(0) == False, -1e10)
        if padding_mask is not None:
            energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == False, -1e10)
        if rg_valid_mask is not None:
            energy = energy.masked_fill(rg_valid_mask.unsqueeze(1).unsqueeze(3) == False, -1e10)

        attention = torch.softmax(energy, dim=-1)  # [T,H,A,G]

        Y1_ = torch.matmul(attention, V)  # [T,H,A,d]
        Y1_ = Y1_.permute(2, 0, 1, 3).contiguous()  # [A,T,H,d]

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + agent
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V  # [A,T,D], [T,H,A,d], [T,H,G,d], [T,H,G,d]

class SelfAttLayer_dec(nn.Module):
    """解码器中的自注意力层。"""
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=head_num)
        self.layer_F1_ = nn.Sequential(
            nn.Linear(feature_dim, k * feature_dim),
            nn.ReLU()
        )
        self.layer_F2_ = nn.Sequential(
            nn.Linear(k * feature_dim, feature_dim),
            nn.ReLU()
        )
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        F, A, T, D = x.shape
        assert (T == self.time_steps and D == self.feature_dim)
        A_check, A_check_ = batch_mask.shape
        A_check_2, T_check = padding_mask.shape
        assert A_check == A and A_check_2 == A and T_check == self.time_steps

        x = self.layer_X_(x)  # [F,A,T,D]

        if self.across_time:
            q = x.reshape(-1, T, D).permute(1, 0, 2)  # [T, F*A, D]
            k, v = q.clone(), q.clone()

            key_padding_mask = padding_mask.repeat(F, 1)  # [F*A, T]
            attn_output, _ = self.layer_att_(q, k, v, key_padding_mask=key_padding_mask)
            att_output = attn_output.permute(1, 0, 2).reshape(F, A, T, D).permute(1, 2, 0, 3)  # [A,T,F,D]
        else:
            q = x.permute(0, 2, 1, 3).reshape(-1, A, D).permute(1, 0, 2)  # [A, F*T, D]
            k, v = q.clone(), q.clone()

            key_padding_mask = padding_mask.permute(1, 0).repeat(F, 1)  # [F*T, A]
            attn_output, _ = self.layer_att_(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
            att_output = attn_output.permute(1, 0, 2).reshape(F, A, T, D).permute(1, 2, 0, 3)  # [A,T,F,D]

        S_ = att_output + x  # 残差连接
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_

def generate_sparse_mask(positions, max_distance=None, time_window=None):
    """
    生成稀疏注意力掩码：
    - 限制代理之间的注意力范围（基于欧几里得距离）。
    - 限制时间步之间的注意力范围（基于滑动时间窗口）。

    Args:
        positions: 代理的位置信息 [A, T, 2]
        max_distance: 限制代理间注意力的最大距离（标量）。
        time_window: 限制时间步注意力的滑动窗口大小（标量）。

    Returns:
        attention_mask: 稀疏注意力掩码 [T, T]。
    """
    A, T, _ = positions.shape

    # 初始化掩码为全连接
    attention_mask = torch.ones(T, T, dtype=torch.bool)

    # 时间步滑动窗口掩码
    if time_window:
        for t in range(T):
            start = max(0, t - time_window)
            end = min(T, t + time_window + 1)
            attention_mask[t, start:end] = False

    return attention_mask  # 转换为 PyTorch MultiheadAttention 的格式（True 表示屏蔽）

def filter_static_agents(positions, threshold=0.5):
    """
    过滤静态代理：
    - 判断代理在所有时间步的最大位移是否小于阈值。

    Args:
        positions: 代理的位置信息 [A, T, 2]
        threshold: 静态代理位移阈值（标量）。

    Returns:
        active_agent_mask: 活动代理的掩码 [A]。
    """
    A, T, _ = positions.shape

    # 计算代理在所有时间步的最大位移
    displacements = torch.norm(positions[:, None, :, :] - positions[:, :, None, :], dim=-1)  # [A, T, T]
    max_displacement = displacements.max(dim=-1)[0].max(dim=-1)[0]  # [A]

    # 标记活动代理（位移大于阈值）
    active_agent_mask = max_displacement > threshold
    return active_agent_mask
