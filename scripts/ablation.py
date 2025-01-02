import time
import torch
import torch.nn as nn
from models.model_sparse_attention import SparseAttentionModel
from models.model_static_filtering import StaticFilteringModel
import matplotlib.pyplot as plt

def test_baseline(batch_size, num_agents, time_steps, feature_dim):
    """
    测试基线模型的计算性能。
    """
    # 数据模拟
    positions = torch.rand(batch_size, num_agents, time_steps, 2) * 100  # 代理随机位置
    features = torch.rand(batch_size, num_agents, time_steps, feature_dim)  # 随机特征

    # 初始化模型
    model = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for b in range(batch_size):
            q = features[b].permute(1, 0, 2)  # [T, A, D]
            k, v = q.clone(), q.clone()
            model(q, k, v)
        elapsed_time = time.time() - start_time

    return elapsed_time

def test_sparse_attention(batch_size, num_agents, time_steps, feature_dim, max_distances, time_window):
    """
    测试稀疏注意力的计算性能随稀疏度变化的影响。
    """
    # 数据模拟
    positions = torch.rand(batch_size, num_agents, time_steps, 2) * 100  # 代理随机位置
    features = torch.rand(batch_size, num_agents, time_steps, feature_dim)  # 随机特征

    # 初始化模型
    model = SparseAttentionModel(feature_dim, time_steps)
    model.eval()

    results = []

    # 测试不同稀疏度
    for max_distance in max_distances:
        with torch.no_grad():
            start_time = time.time()
            for b in range(batch_size):
                model(features[b], positions[b], max_distance, time_window)
            elapsed_time = time.time() - start_time
        results.append(elapsed_time)

    return results

def test_static_filtering(batch_size, num_agents, time_steps, feature_dim, static_thresholds):
    """
    测试静态代理过滤的计算性能随稀疏度变化的影响。
    """
    # 数据模拟
    positions = torch.rand(batch_size, num_agents, time_steps, 2) * 100  # 代理随机位置
    features = torch.rand(batch_size, num_agents, time_steps, feature_dim)  # 随机特征

    # 初始化模型
    model = StaticFilteringModel(feature_dim, time_steps)
    model.eval()

    results = []

    # 测试不同静态阈值
    for threshold in static_thresholds:
        with torch.no_grad():
            start_time = time.time()
            for b in range(batch_size):
                model(features[b], positions[b], threshold)
            elapsed_time = time.time() - start_time
        results.append(elapsed_time)

    return results

def plot_results(x_values, y_values1, y_values2, title, xlabel, ylabel1, ylabel2, output_file):
    """
    绘制折线图展示稀疏度对计算时间的影响和加速比。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel1, color=color, fontsize=14)
    ax1.plot(x_values, y_values1, label="Computation Time", marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(ylabel2, color=color, fontsize=14)
    ax2.plot(x_values, y_values2, label="Speedup", marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # 参数配置
    batch_size = 8
    num_agents = 50
    time_steps = 91
    feature_dim = 256

    # 基线测试
    baseline_time = test_baseline(batch_size, num_agents, time_steps, feature_dim)
    print(f"Baseline computation time: {baseline_time:.4f} seconds")

    # 稀疏注意力测试
    max_distances = [5, 10, 15, 20, 25]  # 距离阈值
    time_window = 10  # 固定时间窗口大小
    sparse_results = test_sparse_attention(batch_size, num_agents, time_steps, feature_dim, max_distances, time_window)
    sparse_speedup = [baseline_time / t for t in sparse_results]
    plot_results(
        max_distances, sparse_results, sparse_speedup,
        title="Sparse Attention Performance and Speedup",
        xlabel="Max Distance (d_max)",
        ylabel1="Computation Time (s)",
        ylabel2="Speedup",
        output_file="./outputs/sparse_attention_performance.pdf"
    )

    # 静态代理过滤测试
    static_thresholds = [0.1, 0.5, 1.0, 1.5, 2.0]  # 静态阈值
    static_results = test_static_filtering(batch_size, num_agents, time_steps, feature_dim, static_thresholds)
    static_speedup = [baseline_time / t for t in static_results]
    plot_results(
        static_thresholds, static_results, static_speedup,
        title="Static Filtering Performance and Speedup",
        xlabel="Static Threshold (δ)",
        ylabel1="Computation Time (s)",
        ylabel2="Speedup",
        output_file="./outputs/static_filtering_performance.pdf"
    )
