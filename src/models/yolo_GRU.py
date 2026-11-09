import torch
import torch.nn as nn
from ultralytics import YOLO

class YoloGRU(nn.Module):
    def __init__(self, num_classes = 3, hidden_size = 256, num_layers=1, pretrained = "yolo26n.pt"):
        super().__init__()

        # Load the YOLO model
        yolo = YOLO(pretrained)
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out