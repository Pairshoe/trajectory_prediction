import torch

# 配置参数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 1000
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 91
NUM_FEATURES = 6
