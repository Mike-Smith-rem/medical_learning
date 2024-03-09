'''Copyright oyk
Created 10 16:46:22
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_segmentation import Down, DoubleConv, Up, OutConv


class UNet_Combine(nn.Module):
    def __init__(self, n_channels, n_class1, n_class2, bilinear=True):
        super(UNet_Combine, self).__init__()
        self.n_channels = n_channels
        self.n_class1 = n_class1
        self.n_class2 = n_class2
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
        self.classifier = nn.Linear(1024 // factor, n_class1)

        # 其他层
        self.up1 = Up(1024, 512 // factor, bilinear)  # in fact is 1024, 256
        self.up2 = Up(512, 256 // factor, bilinear)  # in fact is 512, 128
        self.up3 = Up(256, 128 // factor, bilinear)  # in fact is 256, 64
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_class2)

    def forward(self, x, flag):
        if flag == 0:
            x = self.inc(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # 展平

            # 分类器
            x = self.classifier(x)
            return F.softmax(x, dim=1)
        else:
            x1 = self.inc(x)
            # print("after 3-64 twoConv, x1 shape is: " + str(x1.shape))
            x2 = self.down1(x1)
            # print("after 64-128 max_pool and twoConv, x2 shape is: " + str(x2.shape))
            x3 = self.down2(x2)
            # print("after 128-256 max_pool and twoConv, x3 shape is: " + str(x3.shape))
            x4 = self.down3(x3)
            # print("after 256-512 max_pool and twoConv, x4 shape is: " + str(x4.shape))
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            # print("after 1024-256 max_pool and twoConv, x shape is: " + str(x.shape))
            x = self.up2(x, x3)
            # print("after 512-128 max_pool and twoConv, x shape is: " + str(x.shape))
            x = self.up3(x, x2)
            # print("after 256-64 max_pool and twoConv, x shape is: " + str(x.shape))
            x = self.up4(x, x1)
            # print("after 128-64 max_pool and twoConv, x shape is: " + str(x.shape))
            logits = self.outc(x)
            # print(logits.shape)
            return logits

