import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Bottleneck structure
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Sum the two attention maps
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolution on 2 channels (AvgPool + MaxPool concatenated)
        # Kernel size 7 is standard to capture a wide spatial area
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mean and Max along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention followed by Spatial Attention
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class DoubleConvCBAM(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    Padding=1 is CRUCIAL here to keep the size 256x256
    throughout the network, otherwise the image shrinks at each layer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Helps with faster convergence
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)
        return x


class CBAM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21):
        super(CBAM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER (Contracting Path) ---
        # Increasing the number of filters progressively
        self.inc = DoubleConvCBAM(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConvCBAM(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConvCBAM(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConvCBAM(256, 512))

        # --- BOTTLENECK ---
        # Dropout added to reduce overfitting
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvCBAM(512, 1024),
            nn.Dropout(0.5)
        )

        # --- DECODER (Expansive Path) ---
        # Using ConvTranspose2d for upsampling (inverse operation of convolution)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvCBAM(1024, 512)  # 1024 because we concatenate 512 (skip) + 512 (up)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvCBAM(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConvCBAM(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConvCBAM(128, 64)

        # --- OUTPUT LAYER ---
        # Convolution 1x1 to map to the number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _forward_single(self, x):
        # Encoding and saving residuals (skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding with concatenation
        x = self.up1(x5)
        # We concatenate the output of the upsampling with the encoder residual (x4)
        # This is what gives U-Net its boundary precision
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits

    def forward(self, x):
        # 1. Normal prediction
        logits_normal = self._forward_single(x)

        if not self.training:
            # Horizontal Flip
            x_flip = torch.flip(x, dims=[3])
            
            # Prediction on flipped image
            logits_flip = self._forward_single(x_flip)
            
            # Flip the prediction back
            logits_flip_back = torch.flip(logits_flip, dims=[3])
            
            # Average the two
            return (logits_normal + logits_flip_back) / 2.0
        
        return logits_normal
