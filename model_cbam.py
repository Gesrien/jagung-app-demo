import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, efficientnet_b0, shufflenet_v2_x1_0

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))

# CBAM Module
class CBAM_Module(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM_Module, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# MobileNetV2 with CBAM (CBAM in Depthwise Convolution Layer in Bottleneck)
class MobileNetV2_CBAM(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetV2_CBAM, self).__init__()
        self.mobilenet = mobilenet_v2(weights=None)
        for name, layer in self.mobilenet.features.named_children():
            if isinstance(layer, nn.Conv2d) and layer.groups > 1:  # Depthwise Convolution
                self.mobilenet.features[int(name)] = nn.Sequential(
                    layer,
                    CBAM_Module(layer.out_channels)
                )
        self.mobilenet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.mobilenet.classifier(x)
        return x

# EfficientNetB0 with CBAM (CBAM replaces SE in MBConv)
class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetB0_CBAM, self).__init__()
        self.efficientnet = efficientnet_b0(weights=None)
        for name, layer in self.efficientnet.features.named_children():
            if hasattr(layer, "block_args") and layer.block_args.se_ratio is not None:  # Replace SE with CBAM
                in_channels = layer.conv_pwl.out_channels  # Output channel of MBConv
                self.efficientnet.features[int(name)] = nn.Sequential(
                    layer,
                    CBAM_Module(in_channels)
                )
        self.efficientnet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.efficientnet.classifier(x)
        return x

# ShuffleNetV2 with CBAM (CBAM in Depthwise Convolution in Second Branch of Shuffle Unit)
class ShuffleNetV2_CBAM(nn.Module):
    def __init__(self, num_classes=3):
        super(ShuffleNetV2_CBAM, self).__init__()
        self.shufflenet = shufflenet_v2_x1_0(weights=None)
        for name, layer in self.shufflenet.stage2.named_children():
            if isinstance(layer, nn.Conv2d) and layer.groups > 1:  # Depthwise Convolution
                self.shufflenet.stage2[int(name)] = nn.Sequential(
                    layer,
                    CBAM_Module(layer.out_channels)
                )
        for name, layer in self.shufflenet.stage3.named_children():
            if isinstance(layer, nn.Conv2d) and layer.groups > 1:
                self.shufflenet.stage3[int(name)] = nn.Sequential(
                    layer,
                    CBAM_Module(layer.out_channels)
                )
        self.shufflenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.shufflenet.conv1(x)
        x = self.shufflenet.maxpool(x)
        x = self.shufflenet.stage2(x)
        x = self.shufflenet.stage3(x)
        x = self.shufflenet.stage4(x)
        x = self.shufflenet.conv5(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.shufflenet.fc(x)
        return x
    
# MobileNetV2
class MobileNetV2_NoCBAM(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV2_NoCBAM, self).__init__()
        self.mobilenet = mobilenet_v2(weights=None)
        self.mobilenet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.mobilenet.classifier(x)
        return x

# EfficientNetB0
class EfficientNetB0_NoCBAM(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetB0_NoCBAM, self).__init__()
        self.efficientnet = efficientnet_b0(weights=None)
        self.efficientnet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.efficientnet.classifier(x)
        return x

# ShuffleNetV2
class ShuffleNetV2_NoCBAM(nn.Module):
    def __init__(self, num_classes=8):
        super(ShuffleNetV2_NoCBAM, self).__init__()
        self.shufflenet = shufflenet_v2_x1_0(weights=None)
        self.shufflenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.shufflenet.conv1(x)
        x = self.shufflenet.maxpool(x)
        x = self.shufflenet.stage2(x)
        x = self.shufflenet.stage3(x)
        x = self.shufflenet.stage4(x)
        x = self.shufflenet.conv5(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.shufflenet.fc(x)
        return x
