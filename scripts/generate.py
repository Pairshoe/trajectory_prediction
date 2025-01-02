import torch
from models.predictor import TrajectoryPredictor
from utils.visualization import generate_and_visualize_trajectory
from scripts.config import DEVICE

def main():
    """生成和可视化轨迹。"""
    # 初始化模型
    model = TrajectoryPredictor(hidden_size=64)
    model = model.to(DEVICE)

    # 加载预训练模型
    model.load_state_dict(torch.load("./ckpt/model.pth"))

    # 生成和可视化
    generate_and_visualize_trajectory(model, num_visualizations=3, device=DEVICE)

if __name__ == "__main__":
    main()
