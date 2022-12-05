import torch
import torchvision
import torch.nn as nn
import torch.nn.functional


class BCNN_resnet50_fc(nn.Module):

    def __init__(self, num_classes, pretrained=True):

        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of rsenet18.
        self.features = torchvision.models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-2])  # Remove pooling and fc

        self.fc = torch.nn.Linear(512**2, num_classes)
        
        for param in self.features.parameters(): # Freeze all previous layers.
            param.requires_grad = False
        torch.nn.init.kaiming_normal(self.fc.weight.data)  # Initialize the fc layers.
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)
        self.dropout = nn.Dropout(0.8)

    def forward(self, X):
        
        N = X.size()[0]
        X = self.features(X)
        X = X.view(N, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        #X = self.dropout(X)
        
        X = self.fc(X)
        return X

class BCNN_resnet50_ft(nn.Module):

    def __init__(self, num_classes):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.resnet50(pretrained=False)
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-2])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, num_classes)

        self.dropout = nn.Dropout(0.8)

    def forward(self, X):

        N = X.size()[0]
        X = self.features(X)
        X = X.view(N, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        #X = self.dropout(X)

        X = self.fc(X)
        return X

    # def __init__(self, num_classes, pretrained):
    #     super(BCNN, self).__init__()
    #     features = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    #     # Remove the pooling layer and full connection layer
    #     self.conv = nn.Sequential(*list(features.children())[:-2])
    #     self.fc = nn.Linear(512 * 512, num_classes)
    #     self.softmax = nn.Softmax()
    #     self.dropout = nn.Dropout(0.5)

    #     if pretrained:
    #         for parameter in self.conv.parameters():
    #             parameter.requires_grad = False

    #     nn.init.kaiming_normal_(self.fc.weight.data)
    #     nn.init.constant_(self.fc.bias, val=0)
    #     if self.fc.bias is not None:
    #         torch.nn.init.constant(self.fc.bias.data, val=0)

    # def forward(self, input):
    #     features = self.conv(input)
    #     # Cross product operation
    #     # features.reshape(features.size(0))
    #     features = features.view(features.size(0), 512, 28**2)
    #     features_T = torch.transpose(features, 1, 2)
    #     features = torch.bmm(features, features_T) / (28**2)
    #     features = features.view(features.size(0), 512 * 512)
    #     # The signed square root
    #     features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
    #     # features = torch.sqrt(features + 1e-5)
    #     # L2 regularization
    #     features = torch.nn.functional.normalize(features)

    #     out = self.dropout(features)
    #     out = self.fc(out)

    #     #softmax = self.softmax(out)
    #     #return out, softmax
    #     return out 
    