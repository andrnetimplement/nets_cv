import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class FNC8s(nn.Module):
    def __init__(self, num_classes) -> None:
        super(FNC8s, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.conv1 = vgg.classifier[0]
        self.conv2 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.conv3 = nn.Conv2d(4096, num_classes, kernel_size=1)

        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

        
