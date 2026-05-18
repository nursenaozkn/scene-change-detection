import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

    def forward(self, x):
        f1 = self.enc1(x)              # [B, 32, 512, 512]
        x = self.pool1(f1)             # [B, 32, 256, 256]

        f2 = self.enc2(x)              # [B, 64, 256, 256]
        x = self.pool2(f2)             # [B, 64, 128, 128]

        f3 = self.enc3(x)              # [B, 128, 128, 128]
        x = self.pool3(f3)             # [B, 128, 64, 64]

        bottleneck = self.bottleneck(x) # [B, 256, 64, 64]

        return f1, f2, f3, bottleneck


class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SharedEncoder()

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image_a, image_b):
        a1, a2, a3, ab = self.encoder(image_a)
        b1, b2, b3, bb = self.encoder(image_b)

        d1 = torch.abs(a1 - b1)
        d2 = torch.abs(a2 - b2)
        d3 = torch.abs(a3 - b3)
        db = torch.abs(ab - bb)

        x = self.up3(db)
        x = torch.cat([x, d3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)
