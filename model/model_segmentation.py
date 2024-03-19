'''Copyright oyk
Created 10 14:17:26
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    "Conv => BN => ReLU => Conv => BN => ReLU"
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.ReLU(inplace=True), # inplace: https://discuss.pytorch.org/t/what-is-inplace-operation/16244
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    "MaxPool => DoubleConv"
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.max_pool(x)


class Up(nn.Module):
    "Upsample => DoubleConv"
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # align_corners: https://discuss.pytorch.org/t/what-is-align-corners-true/26745
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print("original_x1: " + str(x1.shape))
        x1 = self.up(x1)
        # print("after_upsample_x1: " + str(x1.shape))
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # print("diffY: " + str(diffY) + " diffX: " + str(diffX))
        # 给x1的上下左右各填充一半的diffY和diffX
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        # print("after_pad_x1: " + str(x1.shape))
        # print("x2: " + str(x2.shape))
        x = torch.cat([x2, x1], dim=1)
        # print("!!!" + str(x.shape))
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear) # in fact is 1024, 256
        self.up2 = Up(512, 256 // factor, bilinear) # in fact is 512, 128
        self.up3 = Up(256, 128 // factor, bilinear) # in fact is 256, 64
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # print("x shape is: " + str(x.shape))
        x1 = self.inc(x)
        # print("after 3-64 twoConv, x1 shape is: " + str(x1.shape))
        x2 = self.down1(x1)
        # print("after 64-128 max_pool and twoConv, x2 shape is: " + str(x2.shape))
        x3 = self.down2(x2)
        # print("after 128-256 max_pool and twoConv, x3 shape is: " + str(x3.shape))
        x4 = self.down3(x3)
        # print("after 256-512 max_pool and twoConv, x4 shape is: " + str(x4.shape))
        x5 = self.down4(x4)
        # print("after 512-512 max_pool and twoConv, x5 shape is: " + str(x5.shape))
        x = self.up1(x5, x4)
        # print("after 1024-256 max_pool and twoConv, x shape is: " + str(x.shape))
        x = self.up2(x, x3)
        # print("after 512-128 max_pool and twoConv, x shape is: " + str(x.shape))
        x = self.up3(x, x2)
        # print("after 256-64 max_pool and twoConv, x shape is: " + str(x.shape))
        x = self.up4(x, x1)
        # print("after 128-64 max_pool and twoConv, x shape is: " + str(x.shape))
        logits = self.outc(x)
        if self.n_classes == 1:
            return F.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)


def freeze_encoder(model, target_model):
    # 冻结加载的参数，不参与训练
    model.inc.load_state_dict(target_model.inc.state_dict())
    for param in model.inc.parameters():
        param.requires_grad = False

    model.down1.load_state_dict(target_model.down1.state_dict())
    for param in model.down1.parameters():
        param.requires_grad = False

    model.down2.load_state_dict(target_model.down2.state_dict())
    for param in model.down2.parameters():
        param.requires_grad = False

    model.down3.load_state_dict(target_model.down3.state_dict())
    for param in model.down3.parameters():
        param.requires_grad = False

    model.down4.load_state_dict(target_model.down4.state_dict())
    for param in model.down4.parameters():
        param.requires_grad = False


if __name__ == "__main__":
    net = UNet(3, 2)
    a = torch.randn(1, 3, 256, 256)
    print(net(a))