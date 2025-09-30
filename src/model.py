import torch
import torch.nn as nn
import torchvision.models as models

class EncoderResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Change first conv layer to accept 1 channel (grayscale)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64
        self.enc3 = resnet.layer2                                   # 128
        self.enc4 = resnet.layer3                                   # 256
        self.enc5 = resnet.layer4                                   # 512

    def forward(self, x):
        x1 = self.enc1(x)  # [B,64,H/2,W/2]
        x2 = self.enc2(x1) # [B,64,H/4,W/4]
        x3 = self.enc3(x2) # [B,128,H/8,W/8]
        x4 = self.enc4(x3) # [B,256,H/16,W/16]
        x5 = self.enc5(x4) # [B,512,H/32,W/32]
        return x1, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNet18_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderResNet18()

        self.dec5 = DecoderBlock(512, 256)
        self.dec4 = DecoderBlock(512, 128)  # skip connection
        self.dec3 = DecoderBlock(256, 64)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Conv2d(128, 1, kernel_size=1)  # grayscale output

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        d5 = self.dec5(x5)
        d4 = self.dec4(torch.cat([d5, x4], dim=1))
        d3 = self.dec3(torch.cat([d4, x3], dim=1))
        d2 = self.dec2(torch.cat([d3, x2], dim=1))
        d1 = self.dec1(torch.cat([d2, x1], dim=1))

        return torch.tanh(d1)  # output [-1,1] matching normalized grayscale
