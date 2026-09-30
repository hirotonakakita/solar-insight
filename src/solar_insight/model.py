import torch
import torch.nn as nn
import torchvision.models as models

class FlarePredictionModel(nn.Module):
    def __init__(self):
        super(FlarePredictionModel, self).__init__()

        # --- CNNバックボーン（簡易ResNet18）---
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()  # 全結合層は除く（512次元特徴量出力）

        # --- 数値特徴用ネットワーク ---
        self.numeric_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # --- 結合後の出力層 ---
        self.combined = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- 出力層（3出力）---
        self.out_m_occurred = nn.Linear(128, 1)  # sigmoid
        self.out_x_occurred = nn.Linear(128, 1)  # sigmoid
        self.out_m_count = nn.Linear(128, 1)     # 回帰

    def forward(self, image, numeric):
        x_img = self.cnn(image)  # 画像特徴（[B, 512]）
        x_num = self.numeric_net(numeric)  # 数値特徴（[B, 64]）

        x = torch.cat((x_img, x_num), dim=1)
        x = self.combined(x)

        m_occurred = torch.sigmoid(self.out_m_occurred(x))
        x_occurred = self.out_x_occurred(x)
        m_count = self.out_m_count(x)  # 回帰なのでそのまま

        return {
            'm_occurred': m_occurred.squeeze(1),
            'x_occurred': x_occurred.squeeze(1),
            'm_count': m_count.squeeze(1)
        }
