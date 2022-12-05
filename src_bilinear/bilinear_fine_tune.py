import torch
import torchvision
import torch.nn as nn
import torch.nn.functional


class BCNN_ft(nn.Module):

    def __init__(self, num_classes):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, num_classes)

        self.dropout = nn.Dropout(0.8)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        #assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        #assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
        #assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        X = self.dropout(X)

        X = self.fc(X)
        #assert X.size() == (N, 200)
        return X