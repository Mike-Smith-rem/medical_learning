import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在分割中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,C,H,W] C表示类别
        :param labels:  实际类别. size:[B,H,W] or [B,1,H,W]
        :return: loss:
        """
        B, C = preds.shape[0], preds.shape[1]
        self.alpha = self.alpha.to(preds.device)  # alpha: [C]

        # 展开
        labels = labels.view(B, -1).long()  # labels:[B, H*W]
        preds = preds.view(B, C, -1)  # preds: [B, C, H*W]
        label_reshaped = labels.detach().view(B, 1, -1)
        preds = torch.gather(preds, dim=1, index=label_reshaped)  # preds: [B, H*W]
        eps = 1e-7  # 防止数值超出定义域
        alpha = self.alpha[labels]  # alpha: [B, H*W]
        # 开始计算
        loss = (-1 * alpha *
                torch.pow((1 - preds), self.gamma) *
                torch.log(preds + eps))

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class BinaryDiceLoss(nn.Module):
    def __init__(self, alpha=None, num_classes=1, size_average=True):
        super(BinaryDiceLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在分割中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        forward
        :param preds: size [B, 2, H, W] or [B, 1, H, W]
        :param labels: size [B, H, W] or [B, 1, H, W]
        :return:
        """
        B, C, H, W = preds.shape
        smooth = 1e-7
        if C == 2:
            preds = F.softmax(preds, dim=1)
            preds = preds[:, 0, :, :]
        probs = preds.view(B, -1)
        labels = labels.view(B, -1)
        loss = torch.zeros(self.num_classes, device=preds.device)
        weight = self.alpha.to(preds.device)

        m1 = probs  # m1: [B, H*W]
        labels = labels
        m2 = labels.view(B, -1)  # m2: [B, H*W]
        intersection = (m1 * m2)
        cls_score = 2. * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
        cls_score = 1 - cls_score
        loss = cls_score * weight

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


if __name__ == '__main__':
    # a = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    # print(F.softmax(a))
    # print(F.softmax(a, dim=0))
    # print(F.softmax(a, dim=1))
    # 创建一个示例输入张量
    # input = torch.tensor([[10, 11, 12]]).contiguous()
    #
    # # 创建一个示例索引张量
    # index = torch.tensor([[2, 1, 0]])
    #
    # # 在 dim=1 维度上使用 gather 函数
    # output = torch.gather(input, dim=1, index=index)
    #
    # print(output)
    # b = torch.tensor([0, 2, 1]).long()
    # print((b == 1).int())
    # a = torch.tensor([
    #     [[0.5, 0.5], [0.9, 0.1], [0.5, 0.6]],
    #     [[0.5, 0.5], [0.9, 0.1], [0.4, 0.5]]
    # ])
    # print(F.sigmoid(a))
    a = torch.tensor([
        [
            [[0.1, 0.6], [0.2, 0.4]],  # h*w, channel 0, b0
            [[0.7, 0.1], [0.6, 0.9]]  # h*w, channel 1, b0
        ],
        [
            [[0.1, 0.6], [0.2, 0.4]],  # h*w, channel 0, b1
            [[0.7, 0.1], [0.6, 0.9]]  # h*w, channel 1, b1
        ]])
    b = torch.tensor(
        [
            [[0, 1], [0, 1]],  # h*w, b0
            [[0, 1], [1, 1]]  # h*w, b1
        ])
    soft_dice_loss = SoftDiceLoss()
    focal_loss = FocalLoss()
    loss1 = soft_dice_loss(a.detach(), b.detach())
    loss2 = focal_loss(a.detach(), b.detach())
    print(loss2)
