import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)


class SiameseChangeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_encoder = SharedEncoder()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, image_a, image_b):
        feat_a = self.shared_encoder(image_a)
        feat_b = self.shared_encoder(image_b)

        diff = torch.abs(feat_a - feat_b)

        out = self.decoder(diff)
        return out