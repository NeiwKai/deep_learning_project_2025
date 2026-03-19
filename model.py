import torch
import torch.nn as nn
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
