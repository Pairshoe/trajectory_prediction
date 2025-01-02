import torch
from torch.utils.data import DataLoader
from datasets.trajectory_dataset import TrajectoryDataset
from models.predictor import TrajectoryPredictor
from scripts.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import torch.nn.functional as F

def train_model(model, train_loader, num_epochs, device, learning_rate):
    """训练模型。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("开始训练模型...")

    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (x, mask) in enumerate(train_loader):
            x = x.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred[mask], x[mask])  # 仅计算未掩盖部分的损失
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f'第 {epoch+1}/{num_epochs} 轮，平均损失: {avg_loss:.6f}')
            loss_values.append(avg_loss)

    # 保存训练损失
    with open('./outputs/loss_values.txt', 'w') as f:
        for loss in loss_values:
            f.write(f"{loss}\n")

    # 绘制训练损失
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(loss_values)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss Value', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('./outputs/loss_values.pdf')

def main():
    """训练主函数。"""
    # 数据集和加载器
    train_dataset = TrajectoryDataset(num_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = TrajectoryPredictor(hidden_size=64)

    # 训练模型
    train_model(model, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE, learning_rate=LEARNING_RATE)

    # 保存模型
    torch.save(model.state_dict(), "./ckpt/model.pth")

if __name__ == "__main__":
    main()
