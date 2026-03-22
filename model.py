import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetMultiHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # backbone (remove classifier)
        self.features = model.features

        # global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = model.classifier[1].in_features
        
        # classification head
        self.cl_head = nn.Linear(in_features, num_classes)

        # bounding box head
        self.bb_head = nn.Linear(in_features, 4)

    def forward(self, x):

        # equivalent to preprocess_input must be done in transforms

        x = self.features(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        
        class_output = torch.softmax(self.cl_head(x), dim=1)
        bbox_output = torch.sigmoid(self.bb_head(x))
        '''
        class_output = self.cl_head(x)
        bbox_output = self.bb_head(x)
        '''

        return class_output, bbox_output

from torchvision.models import resnet18, ResNet18_Weights

class ResNet18MultiHead(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18MultiHead, self).__init__()
        
        # Load ResNet backbone
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fully connected layer

        # The output size of from self.backbone
        n_features = resnet.fc.in_features

        # Classification head
        self.classifier = nn.Linear(n_features, num_classes) 

        # Localization head (bounding box regression)
        self.regressor = nn.Linear(n_features, 4) 

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)  # Flatten the output

        class_out = self.classifier(out) 
        bbox_out = self.regressor(out) 

        return class_out, bbox_out

from torchvision.models import resnet50, ResNet50_Weights

class ResNet50MultiHead(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50MultiHead, self).__init__()

        # Load ResNet50 backbone
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        n_features = resnet.fc.in_features

        self.classifier = nn.Linear(n_features, num_classes)

        self.regressor = nn.Linear(n_features, 4)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out)
        
        return class_out, bbox_out

# -- UNet --

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# Down block
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


# Up block 
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        # Upsample
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


# Full UNet
class UNetMultiHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.down1 = Down(3, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)

        self.bottleneck = DoubleConv(64, 128)

        self.up1 = Up(128 + 64, 64)
        self.up2 = Up(64 + 32, 32)
        self.up3 = Up(32 + 16, 16)
        self.up4 = Up(16 + 8, 8)

        self.classifier = nn.Linear(3276800, num_classes)
        self.regressor = nn.Linear(3276800, 4)

    def forward(self, x):
        # Encoder
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        x = torch.flatten(x, 1)
        class_out = self.classifier(x)
        bbox_out = self.regressor(x)

        return class_out, bbox_out
