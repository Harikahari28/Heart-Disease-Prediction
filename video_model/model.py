import torch
import torch.nn as nn
import torchvision.models.video as video_models

class EchoNetClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.backbone = video_models.r3d_18(pretrained=False)

        in_features = self.backbone.fc.in_features  # 512

        # EXACT classifier head from training
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),          # index 0
            nn.Linear(in_features, 512),  # index 1
            nn.ReLU(inplace=True),        # index 2
            nn.Dropout(dropout * 0.5),    # index 3
            nn.Linear(512, 2)             # index 4
        )

    def forward(self, x):
        return self.backbone(x)
