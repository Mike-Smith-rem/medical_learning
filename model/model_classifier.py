'''Copyright oyk
Created 10 16:46:22
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_segmentation import Down, DoubleConv, Up


class UNet_Classifier(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Classifier, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 下采样部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Linear(1024 // factor, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # 应用全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平

        # 分类器
        x = self.classifier(x)
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    print("!!!")