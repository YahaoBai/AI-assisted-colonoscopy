import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=64):
        super().__init__()
        
        f = base_features
        
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(f * 8, f * 16)
        
        self.att4 = AttentionGate(in_channels=f * 8, gating_channels=f * 16, inter_channels=f * 8)
        self.up4 = UpConv(f * 16, f * 8)
        self.dec4 = ConvBlock(f * 16, f * 8)
        
        self.att3 = AttentionGate(in_channels=f * 4, gating_channels=f * 8, inter_channels=f * 4)
        self.up3 = UpConv(f * 8, f * 4)
        self.dec3 = ConvBlock(f * 8, f * 4)
        
        self.att2 = AttentionGate(in_channels=f * 2, gating_channels=f * 4, inter_channels=f * 2)
        self.up2 = UpConv(f * 4, f * 2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        
        self.att1 = AttentionGate(in_channels=f, gating_channels=f * 2, inter_channels=f)
        self.up1 = UpConv(f * 2, f)
        self.dec1 = ConvBlock(f * 2, f)
        
        self.final = nn.Conv2d(f, out_channels, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        a4 = self.att4(x=e4, g=b)
        d4 = self.up4(b)
        d4 = torch.cat([d4, a4], dim=1)
        d4 = self.dec4(d4)
        
        a3 = self.att3(x=e3, g=d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.dec3(d3)
        
        a2 = self.att2(x=e2, g=d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.dec2(d2)
        
        a1 = self.att1(x=e1, g=d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))


def test():
    x = torch.randn(2, 3, 256, 256)
    model = AttentionUNet(in_channels=3, out_channels=1, base_features=64)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    assert out.shape == x.shape
    print("Test passed!")


if __name__ == "__main__":
    test()
