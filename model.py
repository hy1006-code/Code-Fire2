import torch
import torch.nn as nn
import torch.nn.functional as F


class FireCNN(nn.Module):
    """火情二分类 CNN。

    输入: (B, 3, 128, 128) RGB 图片（train.py 中 resize 到 128）
    结构: 3 组 conv + relu + maxpool（通道数 16 -> 32 -> 64），
          3 次下采样后特征图为 64 x 16 x 16 = 16384，
          再经 fc(16384 -> 256) -> dropout -> fc(256 -> 2) 输出二分类概率。
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # 输入是 3 通道 RGB 图片
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 128x128（padding=1 不改变尺寸）
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 64x64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3 次下采样后: 64 通道 x 16x16 = 16384
        self.fc1 = nn.Linear(16384, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        x = self.conv1(x)     # (B, 16, 128, 128)
        x = F.relu(x)
        x = self.pool(x)      # (B, 16, 64, 64)

        x = self.conv2(x)     # (B, 32, 64, 64)
        x = F.relu(x)
        x = self.pool(x)      # (B, 32, 32, 32)

        x = self.conv3(x)     # (B, 64, 32, 32)
        x = F.relu(x)
        x = self.pool(x)      # (B, 64, 16, 16)

        x = torch.flatten(x, start_dim=1)  # (B, 16384)
        x = F.relu(self.fc1(x))            # (B, 256)
        x = self.dropout(x)
        return self.fc2(x)                 # (B, 2)
