import torch
from torch import nn
import torchvision

class VGGLoss(nn.Module):
    def __init__(self, feature_layers=36):
        # VGG19 has 0-36 indexed feature layers, Last layer being MaxPool2d
        super(VGGLoss, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        vgg_feature_layers = nn.Sequential(*list(self.vgg19.features)[:feature_layers]).eval()
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(self.vgg19(input), self.vgg19(target))
