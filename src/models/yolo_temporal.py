import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Literal, Optional

class YoloTemporal(nn.Module):
    def __init__(self,feature_dim, num_classes = 10, hidden_size = 256, num_layers=2, dropout = 0.3, bidirectional = False, model_type: Literal["gru", "lstm"] = "gru"):
        super().__init__()
        
        if feature_dim <= 0 or num_classes <= 0 or num_layers <= 0:
            raise ValueError("feature_dim, num_classes, num_layers must be positive")
        if model_type not in ("gru", "lstm"):
            raise ValueError("model_type must be either 'gru' or 'lstm'")
        
        self.feature_dim = feature_dim
        self.model_type = model_type
        self.bidirectional = bidirectional
        recurrent_cls = nn.GRU if model_type == "gru" else nn.LSTM
        
        self.recurrent = recurrent_cls(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=bidirectional)
        
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, features: Tensor, lengths: Optional[Tensor] = None):
        ## validate video clip data
        if features.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {features.shape[-1]}")
        if features.shape[1] == 0:
            raise ValueError("Sequences cannot be empty")
        
        if lengths is not None:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
            if lengths.ndim != 1 or lengths.numel() != features.shape[0]:
                raise ValueError("lengths must be [B]")
            if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
                raise ValueError("Each length must be between 1 and T")
            sequence = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        else:
            sequence = features
            
        _, hidden = self.recurrent(sequence)
        if self.model_type == "lstm":
            hidden = hidden[0]  # LSTM returns (hidden_state, cell_state)
            
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            last_hidden = hidden[-1]
        return self.classifier(last_hidden)

class GRUEventClassifier(YoloTemporal):
    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(feature_dim=feature_dim, model_type="gru", **kwargs)


class LSTMEventClassifier(YoloTemporal):
    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(feature_dim=feature_dim, model_type="lstm", **kwargs)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch = torch.randn(4, 16, 512)
    labels = torch.tensor([0, 1, 2, 3])
    for cls in (GRUEventClassifier, LSTMEventClassifier):
        model = cls(feature_dim=512, num_classes=10)
        logits = model(batch)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        print(cls.__name__, "logits", tuple(logits.shape), "loss", round(loss.item(), 4))