import torch
import pytorch_lightning as pl
from .encoder import Encoder
from .decoder import Decoder

class SceneTransformer(pl.LightningModule):
    """主模型模块。"""
    def __init__(self, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, 
                 time_steps=91, feature_dim=256, head_num=4, k=4, num_features=6):
        super().__init__()
        self.encoder = Encoder(in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, 
                               time_steps, feature_dim, head_num, k=k)
        self.decoder = Decoder(time_steps, feature_dim, head_num, k, num_features)

    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, 
                states_hidden_mask_batch, roadgraph_feat_batch, roadgraph_valid_batch,
                traffic_light_feat_batch, traffic_light_valid_batch, agent_rg_mask, agent_traffic_mask):
        encodings = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, 
                                 states_hidden_mask_batch, roadgraph_feat_batch, roadgraph_valid_batch,
                                 traffic_light_feat_batch, traffic_light_valid_batch, agent_rg_mask, agent_traffic_mask)
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch)
        return decoding
