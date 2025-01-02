import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def generate_and_visualize_trajectory(model, num_visualizations=1, device='cpu'):
    """生成并可视化轨迹。"""
    model.eval()
    from datasets.trajectory_dataset import TrajectoryDataset

    dataset = TrajectoryDataset(num_samples=num_visualizations)

    for i in range(num_visualizations):
        trajectory, _ = dataset[i]

        with torch.no_grad():
            trajectory_input = trajectory.unsqueeze(0).to(device)
            pred = model(trajectory_input).squeeze(0).cpu().numpy()

        trajectory = trajectory.numpy()

        plt.figure(figsize=(12, 9))
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c='blue', s=10, alpha=0.5, label='Actual trajectory')
        plt.scatter(pred[:, 0], pred[:, 1], c='gold', s=10, alpha=0.5, label='Predicted trajectory')
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
        plt.plot(pred[:, 0], pred[:, 1], 'r-', alpha=0.3)
        plt.gca().set_facecolor('indigo')
        plt.grid(False)

        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks()
        plt.yticks()
        plt.gca().invert_yaxis()
        plt.tight_layout()

        np.savetxt('./outputs/trajectory_data_{}.txt'.format(i), np.concatenate((trajectory, pred), axis=1), delimiter=',')
        plt.savefig('./outputs/trajectory_{}.pdf'.format(i))
        plt.close()
        print('Finished visualization {}.'.format(i))
