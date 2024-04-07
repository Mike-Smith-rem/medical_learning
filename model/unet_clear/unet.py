import torch
import torch.nn as nn

# from resnet import resnet50
from .vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained = pretrained)
            in_filters = [192, 384, 768, 1024]
        # elif backbone == "resnet50":
        #     self.resnet = resnet50(pretrained = pretrained)
        #     in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 32,32,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 64,64,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 128,128,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 256,256,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, type):
        inputs = inputs.to(torch.float32)
        if type == 'cls':
            # [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
            # FIXME
            out = self.vgg.forward(inputs, type)
            return out
        else:
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs, type)
        # elif self.backbone == "resnet50":
        #     [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
            up4 = self.up_concat4(feat4, feat5)
            up3 = self.up_concat3(feat3, up4)
            up2 = self.up_concat2(feat2, up3)
            up1 = self.up_concat1(feat1, up2)

            # 判断有没有up_conv这个层
            if self.up_conv is not None:
                up1 = self.up_conv(up1)
            # 输入最后一层，得到分割结果
            final = self.final(up1)
            # print("final:")
            # print(final)
            return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
