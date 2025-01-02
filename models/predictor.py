import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    """轨迹预测器。"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, 
                           num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions
