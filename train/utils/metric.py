# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import torch
import numpy as np
import torch.nn.functional as F


def binary_dice_score(label_gt, label_pred, threshold: float = 0.5) -> dict:
    """
    :param threshold:
    :param label_gt: [B, H, W] (2D images) or [B, 1, H, W] (ground truth)
    :param label_pred: [B, H, W] (2D images) or [B, 1, H, W] (predicted)
    :return: dict with dice score
    """
    epsilon = 1.0e-6
    B, C = label_pred.shape[0], label_pred.shape[1]
    # print(label_pred.shape)
    label_gt = label_gt.detach().view(B, -1)
    label_pred = label_pred.detach().view(B, C, -1)
    if C != 1:
        label_pred = torch.argmax(label_pred, dim=1)
    else:
        label_pred = label_pred.view(B, -1)
        label_pred = (label_pred > threshold).int()

    # label_pred = (label_pred > threshold).int()
    # print(label_gt)
    # print(label_pred)
    # if label_gt.max() == 0:
    #     label_gt = 1 - label_gt
    #     label_pred = 1 - label_pred
    score = (2.0 * torch.sum(label_gt * label_pred) + epsilon) / (torch.sum(label_gt) + torch.sum(label_pred) + epsilon)
    return {'dice': score}


def multi_dice_score(label_gt, label_pred, threshold: float = 0.5) -> dict:
    """
    Overview: multi-label dice score, C is the number of classes with 0 as background
    :param threshold: the threshold of [0, 1]
    :param label_gt: size [B, H, W] or [B, 1, H, W] (ground truth)
    :param label_pred: size [B, C, H, W]
    :return:
    """
    epsilon = 1.0e-6
    B, C, H, W = label_pred.shape
    # label_pred = torch.argmax(label_pred, dim=1).detach()
    dice_scores = torch.zeros(C, dtype=torch.float32)
    label_gt = label_gt.view(B, -1)  # label_gt: [B, H*W]
    label_pred = label_pred.view(B, C, -1)
    label_pred = (label_pred > threshold).int()

    for index_id in range(C):
        class_id = index_id + 1
        c_label_pred = label_pred[:, index_id].detach().view(B, -1)
        img_A = (label_gt == class_id).int().flatten()
        img_B = (c_label_pred == class_id).int().flatten()
        score = 2.0 * torch.sum(img_A * img_B) / (torch.sum(img_A) + torch.sum(img_B) + epsilon)
        dice_scores[index_id] = score

    return {'mean_dice': dice_scores.mean(), 'dice': dice_scores}


def acc_scores(label_gt, label_pred) -> dict:
    '''
    overview: classification, C is the number of classes with 0 important
    :param label_gt: size [B]
    :param label_pred: [B, C]
    :return:
    '''
    B, C = label_pred.shape[0], label_pred.shape[1]
    label_pred = torch.argmax(label_pred, dim=1)
    eps = 1e-6
    accuracy = torch.zeros(C, dtype=torch.float32)
    for class_id in range(C):
        # print(f"class {class_id}")
        gt = (label_gt == class_id).int().flatten()
        pred = (label_pred == class_id).int().flatten()
        accuracy[class_id] = ((pred * gt).sum()) / (gt.sum() + pred.sum() - (pred * gt).sum() + eps)
    acc_total = (label_pred == label_gt).int().sum() / torch.ones(label_gt.shape).sum()
    # print("end acc count..")
    return {
        'mean_acc': acc_total,
        'acc': accuracy
    }


def f_score(inputs,target,beta=1,smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    # temp_inputs is n*（h*w）*c,每个值是第c类得概率
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_inputs = temp_inputs.argmax(dim=-1).flatten()
    temp_target = target.flatten()
    intersection = torch.sum(temp_inputs * temp_target)
    dice_score = (2. * intersection +smooth)/ (torch.sum(temp_inputs) + torch.sum(temp_target)+smooth)
    return dice_score


# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
# reference from https://blog.csdn.net/weixin_43143670/article/details/104791946
def _fast_hist(label_true, label_pred, n_class):
    '''
    overview: classification matrixs. returns a tensor
    :param label_true:  size [B,]
    :param label_pred:  size [B,](already use argmax)
    :param n_class: the class number
    :return: a tensor shaped of [C, C], where c is the class number
    '''
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * torch.tensor(label_true[mask], dtype=torch.int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    '''
    overview: return the overall acc and iu
    :param label_trues: shape [B,]
    :param label_preds: shape [B, C]
    :param n_class: C
    :return: total_acc, mean_acc, mean_iu
    '''
    hist = torch.zeros((n_class, n_class))
    # FIXME: there is no need to segment the B dimension
    # for lt, lp in zip(label_trues, label_preds):
    hist += _fast_hist(label_trues.flatten(), label_preds.flatten(), n_class)
    print(hist)
    acc = torch.diag(hist).sum() / hist.sum()
    with torch.no_grad():
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
    acc_cls = np.nanmean(acc_cls.detach().numpy())
    with torch.no_grad():
        iu = torch.diag(hist) / (
                hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)
        )
    mean_iu = np.nanmean(iu.detach().numpy())
    freq = hist.sum(dim=1) / hist.sum()
    return acc, acc_cls, mean_iu


if __name__ == "__main__":
    # a = torch.tensor([1, 2, 0, 2, 2, 0, 1, 1, 0])
    # b = torch.tensor([1, 2, 0, 2, 2, 0, 1, 1, 1])
    # b = torch.tensor([[0.1, 0.6, 0.3],  # 1
    #                   [0.1, 0.2, 0.7],  # 2
    #                   [0.8, 0.1, 0.1],  # 0
    #                   [0.1, 0.3, 0.6],  # 2
    #                   [0.3, 0.1, 0.6],  # 2
    #                   [0.5, 0.2, 0.3],  # 0
    #                   [0.1, 0.6, 0.3],  # 1
    #                   [0.1, 0.6, 0.3],  # 1
    #                   [0.1, 0.6, 0.3]])  # 1
    # acc, acc_cls, mean_iu = label_accuracy_score(a, torch.argmax(b, dim=1), 3)
    a = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    b = torch.tensor([
                    [
                        [[0.6, 0.4, 0.6],
                        [0.9, 0.1, 0.1],
                        [0.9, 0.9, 0.9]],
                        [[0.4, 0.6, 0.4],
                         [0.1, 0.9, 0.9],
                         [0.1, 0.1, 0.1]],
                    ],
                    [
                        [[0.6, 0.4, 0.6],
                         [0.9, 0.1, 0.1],
                         [0.9, 0.9, 0.9]],
                        [[0.4, 0.6, 0.4],
                         [0.1, 0.9, 0.9],
                         [0.1, 0.1, 0.1]],
                    ]
    ])
    print(binary_dice_score(a, b))
