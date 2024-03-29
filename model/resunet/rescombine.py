'''Copyright oyk
Created 28 16:57:40
'''
import torch
import torch.nn as nn

from .resnet_blocks import BasicBlock
from .resunet import _resnet
from .unet_blocks import UNetUpBlock


class Res50_Unet_Combine(nn.Module):

    def __init__(self, class_seg=2, class_cls=3, base_channels=64, b_RGB=True,
                 level=5, padding=1, norm_layer=None, bilinear=True, pretrained=False):
        super(Res50_Unet_Combine, self).__init__()
        self.base_channels = base_channels
        self.level = level
        self.padding = padding
        self.norm_layer = norm_layer
        self.bilinear = bilinear
        self.class_seg = class_seg
        self.class_cls = class_cls
        self.b_RGB = b_RGB
        self.pretrained = pretrained

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.classifier = self.build_classifier(class_cls)

    def build_encoder(self):
        return _resnet('resnet50', BasicBlock, [3, 4, 6, 3], base_planes=self.base_channels, pretrained=self.pretrained)

    def build_decoder(self):
        decoder = nn.ModuleList()
        for i in range(self.level - 1):  # 有 self.level-1 个上采样块
            in_channels = self.base_channels * (2 ** (self.level - i - 1))
            out_channels = self.base_channels * (2 ** (self.level - i - 2))
            decoder.append(UNetUpBlock(in_channels, out_channels,
                                       padding=self.padding, norm_layer=self.norm_layer, bilinear=self.bilinear))
        # FIXME: If the out block is neccessary, you can just put it in the init()
        outBlock = nn.Sequential(nn.Conv2d(self.base_channels, self.class_seg, 1, 1), nn.Sigmoid())
        decoder.append(outBlock)
        return decoder

    def build_classifier(self, num_classes):
        classifier = nn.ModuleList()
        # just aasert the classifier contains the avg and fc
        # if self.include_top:
        avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
        fc = nn.Linear(self.base_channels * (2 ** (self.level - 1)), num_classes)
        classifier.append(avgpool)
        classifier.append(fc)

        return classifier

    def forward(self, x, type):
        features = self.encoder(x)[0:self.level]
        # print('encoder feature: {}'.format(features[-2].shape))
        # for feat in features:
        #     print(feat.shape)
        assert len(features) == self.level
        x = features[-1]
        if type == 'cls':
            avg = self.classifier[0]
            fc = self.classifier[1]
            x = avg(x)
            x = torch.flatten(x, 1)
            x = fc(x)
            return x
        else:
            # for i, up_block in enumerate(self.decoder):
            #     x = up_block(x, features[-2 - i])
            #     # print("shape:{}".format(x.shape))
            # if self.outBlock is not None:
            #     x = self.outBlock(x)
            # 加一个softmax激活函数 或则sigmoid也行
            for i in range(len(features) - 1):
                decoder_item = self.decoder[i]
                if isinstance(decoder_item, UNetUpBlock):
                    x = decoder_item(x, features[-2 - i])
                    # print("shape:{}".format(x.shape))
                else:
                    raise Exception('decoder type is {} of index {} '
                                    .format(str(type(decoder_item)), i))
            out_block = self.decoder[-1]
            x = out_block(x)
            return x


if __name__ == '__main__':
    net = Res50_Unet_Combine()
    # print(net)
    # net2 = torchvision.models.resnet50()
    # print(net2)
    img = torch.rand([1, 3, 256, 256])
    output = net(img, 'seg')
    # print(output.shape)
    # output = net(img, 'cls')
    # print(output.shape)