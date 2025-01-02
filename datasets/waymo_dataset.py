import numpy as np
import torch
from torch.utils.data import Dataset

class WaymoDataset(Dataset):
    """模拟的Waymo数据集。"""
    def __init__(self, num_samples=1000):
        super().__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟数据
        data = {}

        # 道路图特征
        data['roadgraph_samples/dir'] = np.random.rand(20000, 3).astype(np.float32)
        data['roadgraph_samples/id'] = np.random.randint(0, 1000, size=(20000, 1)).astype(np.int32)
        data['roadgraph_samples/type'] = np.random.randint(0, 5, size=(20000, 1)).astype(np.int32)
        data['roadgraph_samples/valid'] = np.random.randint(0, 2, size=(20000, 1)).astype(np.int32)
        data['roadgraph_samples/xyz'] = np.random.rand(20000, 3).astype(np.float32)

        # 状态特征
        data['state/id'] = np.random.rand(128).astype(np.float32)
        data['state/type'] = np.random.rand(128).astype(np.float32)
        data['state/is_sdc'] = np.random.randint(0, 2, size=(128,)).astype(np.int32)
        data['state/tracks_to_predict'] = np.random.randint(0, 2, size=(128,)).astype(np.int32)

        # 当前状态
        data['state/current/bbox_yaw'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/height'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/length'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/timestamp_micros'] = np.random.randint(1e12, 1e13, size=(128,1)).astype(np.int64)
        data['state/current/valid'] = np.random.randint(0, 2, size=(128,1)).astype(np.int32)
        data['state/current/vel_yaw'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/velocity_x'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/velocity_y'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/width'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/x'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/y'] = np.random.rand(128, 1).astype(np.float32)
        data['state/current/z'] = np.random.rand(128, 1).astype(np.float32)

        # 未来状态
        data['state/future/bbox_yaw'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/height'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/length'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/timestamp_micros'] = np.random.randint(1e12, 1e13, size=(128,80)).astype(np.int64)
        data['state/future/valid'] = np.random.randint(0, 2, size=(128,80)).astype(np.int32)
        data['state/future/vel_yaw'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/velocity_x'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/velocity_y'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/width'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/x'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/y'] = np.random.rand(128, 80).astype(np.float32)
        data['state/future/z'] = np.random.rand(128, 80).astype(np.float32)

        # 过去状态
        data['state/past/bbox_yaw'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/height'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/length'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/timestamp_micros'] = np.random.randint(1e12, 1e13, size=(128,10)).astype(np.int64)
        data['state/past/valid'] = np.random.randint(0, 2, size=(128,10)).astype(np.int32)
        data['state/past/vel_yaw'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/velocity_x'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/velocity_y'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/width'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/x'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/y'] = np.random.rand(128, 10).astype(np.float32)
        data['state/past/z'] = np.random.rand(128, 10).astype(np.float32)

        # 交通灯特征
        # 当前状态
        data['traffic_light_state/current/state'] = np.random.randint(0, 10, size=(1,16)).astype(np.int32)
        data['traffic_light_state/current/valid'] = np.random.randint(0, 2, size=(1,16)).astype(np.int32)
        data['traffic_light_state/current/x'] = np.random.rand(1,16).astype(np.float32)
        data['traffic_light_state/current/y'] = np.random.rand(1,16).astype(np.float32)
        data['traffic_light_state/current/z'] = np.random.rand(1,16).astype(np.float32)

        # 过去状态
        data['traffic_light_state/past/state'] = np.random.randint(0, 10, size=(10,16)).astype(np.int32)
        data['traffic_light_state/past/valid'] = np.random.randint(0, 2, size=(10,16)).astype(np.int32)
        data['traffic_light_state/past/x'] = np.random.rand(10,16).astype(np.float32)
        data['traffic_light_state/past/y'] = np.random.rand(10,16).astype(np.float32)
        data['traffic_light_state/past/z'] = np.random.rand(10,16).astype(np.float32)

        # 未来状态
        data['traffic_light_state/future/state'] = np.random.randint(0, 10, size=(80,16)).astype(np.int32)
        data['traffic_light_state/future/valid'] = np.random.randint(0, 2, size=(80,16)).astype(np.int32)
        data['traffic_light_state/future/x'] = np.random.rand(80,16).astype(np.float32)
        data['traffic_light_state/future/y'] = np.random.rand(80,16).astype(np.float32)
        data['traffic_light_state/future/z'] = np.random.rand(80,16).astype(np.float32)

        return data

def waymo_collate_fn(batch, GD=16, GS=1400): 
    """自定义的批处理函数。"""
    past_states_batch = np.array([]).reshape(-1,10,9)
    past_states_valid_batch = np.array([]).reshape(-1,10)
    current_states_batch = np.array([]).reshape(-1,1,9)
    current_states_valid_batch = np.array([]).reshape(-1,1)
    future_states_batch = np.array([]).reshape(-1,80,9)
    future_states_valid_batch = np.array([]).reshape(-1,80)
    states_batch = np.array([]).reshape(-1,91,9)

    states_padding_mask_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_BP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_CBP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_GDP_batch = np.array([]).reshape(-1,91)

    roadgraph_feat_batch = np.array([]).reshape(-1,91,6)
    roadgraph_valid_batch = np.array([]).reshape(-1,91)

    traffic_light_feat_batch = np.array([]).reshape(-1,91,3)
    traffic_light_valid_batch = np.array([]).reshape(-1,91)

    num_agents = np.array([])

    for data in batch:
        # 代理状态
        past_states = np.stack((data['state/past/x'],data['state/past/y'],data['state/past/bbox_yaw'],
                                    data['state/past/velocity_x'],data['state/past/velocity_y'],data['state/past/vel_yaw'],
                                        data['state/past/width'],data['state/past/length'],data['state/past/timestamp_micros']), axis=-1)
        past_states_valid = data['state/past/valid'] > 0.
        current_states = np.stack((data['state/current/x'],data['state/current/y'],data['state/current/bbox_yaw'],
                                    data['state/current/velocity_x'],data['state/current/velocity_y'],data['state/current/vel_yaw'],
                                        data['state/current/width'],data['state/current/length'],data['state/current/timestamp_micros']), axis=-1)
        current_states_valid = data['state/current/valid'] > 0.
        future_states = np.stack((data['state/future/x'],data['state/future/y'],data['state/future/bbox_yaw'],
                                    data['state/future/velocity_x'],data['state/future/velocity_y'],data['state/future/vel_yaw'],
                                        data['state/future/width'],data['state/future/length'],data['state/future/timestamp_micros']), axis=-1)
        future_states_valid = data['state/future/valid'] > 0.

        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)
        states_valid = np.concatenate((past_states_valid,current_states_valid,future_states_valid),axis=1)
        states_any_mask = np.sum(states_valid,axis=1) > 0
        states_feat = states_feat[states_any_mask]

        states_padding_mask = np.concatenate((past_states_valid[states_any_mask],current_states_valid[states_any_mask],future_states_valid[states_any_mask]), axis=1)
        
        # 创建隐藏掩码
        states_hidden_mask_BP = np.ones((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_BP[:,:12] = False
        sdc_indices = np.where(data['state/is_sdc'][states_any_mask] == 1)[0]
        if len(sdc_indices) > 0:
            sdvidx = sdc_indices[0]
        else:
            sdvidx = 0
        states_hidden_mask_CBP = np.zeros((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_CBP[:,:12] = False
        if sdvidx < len(states_hidden_mask_CBP):
            states_hidden_mask_CBP[sdvidx,:] = False
        states_hidden_mask_GDP = np.zeros((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_GDP[:,:12] = False
        if sdvidx < len(states_hidden_mask_GDP):
            states_hidden_mask_GDP[sdvidx,-1] = False

        num_agents = np.append(num_agents, len(states_feat))
        
        # 静态道路图
        roadgraph_feat = np.concatenate((data['roadgraph_samples/id'], data['roadgraph_samples/type'], 
                                            data['roadgraph_samples/xyz'][:,:2], data['roadgraph_samples/dir'][:,:2]), axis=-1)
        roadgraph_valid = data['roadgraph_samples/valid'] > 0.
        valid_num = roadgraph_valid.sum()
        if valid_num > GS:
            roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
            spacing = valid_num // GS
            roadgraph_feat = roadgraph_feat[::spacing, :]
            remove_num = len(roadgraph_feat) - GS
            if remove_num > 0:
                roadgraph_mask2 = np.full(len(roadgraph_feat), True)
                idx_remove = np.random.choice(range(len(roadgraph_feat)), remove_num, replace=False)
                roadgraph_mask2[idx_remove] = False
                roadgraph_feat = roadgraph_feat[roadgraph_mask2]
                roadgraph_valid = np.ones((GS,1)).astype(np.bool_)
        else:
            roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]

            roadgraph_valid = np.zeros((GS,1)).astype(np.bool_)
            roadgraph_valid[:valid_num,:] = True

        roadgraph_feat = np.repeat(roadgraph_feat[:,np.newaxis,:],91,axis=1)
        roadgraph_valid = np.repeat(roadgraph_valid,91,axis=1)


        # 拼接批次
        past_states_batch = np.concatenate((past_states_batch, past_states), axis=0)
        past_states_valid_batch = np.concatenate((past_states_valid_batch, past_states_valid), axis=0)
        current_states_batch = np.concatenate((current_states_batch, current_states), axis=0)
        current_states_valid_batch = np.concatenate((current_states_valid_batch, current_states_valid), axis=0)
        future_states_batch = np.concatenate((future_states_batch, future_states), axis=0)
        future_states_valid_batch = np.concatenate((future_states_valid_batch, future_states_valid), axis=0)

        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_mask_batch = np.concatenate((states_padding_mask_batch,states_padding_mask), axis=0)

        states_hidden_mask_BP_batch = np.concatenate((states_hidden_mask_BP_batch,states_hidden_mask_BP), axis=0)
        states_hidden_mask_CBP_batch = np.concatenate((states_hidden_mask_CBP_batch,states_hidden_mask_CBP), axis=0)
        states_hidden_mask_GDP_batch = np.concatenate((states_hidden_mask_GDP_batch,states_hidden_mask_GDP), axis=0)

        roadgraph_feat_batch = np.concatenate((roadgraph_feat_batch, roadgraph_feat), axis=0)
        roadgraph_valid_batch = np.concatenate((roadgraph_valid_batch, roadgraph_valid), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)
    agents_batch_mask = np.zeros((num_agents_accum[-1],num_agents_accum[-1]))
    agent_rg_mask = np.zeros((num_agents_accum[-1],len(num_agents)*GS))
    agent_traffic_mask = np.zeros((num_agents_accum[-1],len(num_agents)*GD))

    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 1
        agent_rg_mask[num_agents_accum[i]:num_agents_accum[i+1], GS*i:GS*(i+1)] = 1
        agent_traffic_mask[num_agents_accum[i]:num_agents_accum[i+1], GD*i:GD*(i+1)] = 1

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_mask_batch = torch.BoolTensor(states_padding_mask_batch)
    states_hidden_mask_BP_batch = torch.BoolTensor(states_hidden_mask_BP_batch)
    states_hidden_mask_CBP_batch = torch.BoolTensor(states_hidden_mask_CBP_batch)
    states_hidden_mask_GDP_batch = torch.BoolTensor(states_hidden_mask_GDP_batch)
    
    roadgraph_feat_batch = torch.FloatTensor(roadgraph_feat_batch)
    roadgraph_valid_batch = torch.BoolTensor(roadgraph_valid_batch)
    traffic_light_feat_batch = torch.FloatTensor(traffic_light_feat_batch)
    traffic_light_valid_batch = torch.BoolTensor(traffic_light_valid_batch)

    agent_rg_mask = torch.BoolTensor(agent_rg_mask)
    agent_traffic_mask = torch.BoolTensor(agent_traffic_mask)

    return (states_batch, agents_batch_mask, states_padding_mask_batch, 
                (states_hidden_mask_BP_batch, states_hidden_mask_CBP_batch, states_hidden_mask_GDP_batch), 
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                        agent_rg_mask, agent_traffic_mask)