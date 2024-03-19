'''Copyright oyk
Created 10 16:46:22
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_segmentation import Down, DoubleConv, Up, OutConv


class UnetEncoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UnetEncoder, self).__init__()
        self.n_channels = n_channels

        # 下采样部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UnetClassifier(nn.Module):
    def __init__(self, n_class, bilinear=True):
        super(UnetClassifier, self).__init__()
        self.n_class = n_class
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        factor = 2 if bilinear else 1
        # 分类器
        self.classifier = nn.Linear(1024 // factor, n_class)

    def forward(self, x):
        x = self.avg_pool(x[-1])
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class UnetDecoder(nn.Module):
    def __init__(self, n_class, bilinear=True):
        super(UnetDecoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)  # in fact is 1024, 256
        self.up2 = Up(512, 256 // factor, bilinear)  # in fact is 512, 128
        self.up3 = Up(256, 128 // factor, bilinear)  # in fact is 256, 64
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_class)

    def forward(self, x):
        xt = self.up1(x[-1], x[-2])
        xt = self.up2(xt, x[-3])
        xt = self.up3(xt, x[-4])
        xt = self.up4(xt, x[-5])
        logits = self.outc(xt)
        return F.sigmoid(logits)


class UnetCombine(nn.Module):
    def __init__(self, n_channels, n_class1, n_class2, bilinear=True):
        super(UnetCombine, self).__init__()
        self.n_channels = n_channels
        self.n_class1 = n_class1
        self.n_class2 = n_class2
        self.bilinear = bilinear

        # 下采样部分
        self.encoder = UnetEncoder(n_channels)
        self.classifier = UnetClassifier(n_class1)
        self.decoder = UnetDecoder(n_class2)

    def forward(self, x, flag=None):
        x1, x2, x3, x4, x5 = self.encoder(x)
        xt = (x1, x2, x3, x4, x5)
        if flag == 'cls':
            return self.classifier(xt)
        else:
            return self.decoder(xt)


class UnetCls(nn.Module):
    def __init__(self, encoder, decoder):
        super(UnetCls, self).__init__()
        self.encoder = encoder
        self.classifier = decoder

    def forward(self, x):
        return self.classifier(self.encoder(x))


class UnetSeg(nn.Module):
    def __init__(self, encoder, decoder):
        super(UnetSeg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))