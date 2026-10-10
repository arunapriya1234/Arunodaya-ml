import torch
import torch.nn as nn

class LandmarkMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for landmark-based gesture and mudra classification.
    Supports single-hand (63-dim) or dual-hand (126-dim) input representations.
    Outputs logits across arbitrary number of gesture/mudra classes.
    """
    def __init__(self, input_dim: int = 126, hidden_dim: int = 256, num_classes: int = 47, dropout: float = 0.25):
        super(LandmarkMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
