# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import torch


def binary_dice_score(label_gt, label_pred, threshold: float = 0.5) -> dict:
    """
    :param threshold:
    :param label_gt: [B, H, W] (2D images) or [B, 1, H, W] (ground truth)
    :param label_pred: [B, H, W] (2D images) or [B, 1, H, W] (predicted)
    :return: dict with dice score
    """
    epsilon = 1.0e-6
    B, C = label_pred.shape[0], label_pred.shape[1]
    label_gt = label_gt.detach().view(B, -1)
    label_pred = torch.argmax(label_pred, dim=1)
    label_pred = label_pred.detach().view(B, -1)

    label_pred = (label_pred > threshold).int()
    if label_gt.max() == 0:
        label_gt = 1 - label_gt
        label_pred = 1 - label_pred
    score = 2.0 * torch.sum(label_gt * label_pred) / (torch.sum(label_gt) + torch.sum(label_pred) + epsilon)
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
    label_gt = label_gt.view(B, -1)        # label_gt: [B, H*W]
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
    # print("start acc count..")
    for class_id in range(C):
        # print(f"class {class_id}")
        gt = (label_gt == class_id).int().flatten()
        pred = (label_pred == class_id).int().flatten()
        accuracy[class_id] = ((pred * gt).sum()) / (gt.sum() + ((1 - pred) * gt).sum() + eps)
    # print("end acc count..")
    return {
        'mean_acc': accuracy.mean(),
        'acc': accuracy
    }



if __name__ == "__main__":
    a = torch.tensor([1, 2, 0, 2, 2, 0, 1, 1, 0])
    b = torch.tensor([1, 2, 0, 2, 2, 0, 1, 1, 1])
    b = torch.tensor([[0.1, 0.6, 0.3], # 1
                      [0.1, 0.2, 0.7], # 1
                      [0.8, 0.1, 0.1], # 0
                      [0.1, 0.3, 0.6], # 2
                      [0.3, 0.1, 0.6], # 2
                      [0.5, 0.2, 0.3], # 0
                      [0.1, 0.6, 0.3], # 1
                      [0.1, 0.6, 0.3], # 1
                      [0.1, 0.6, 0.3]]) # 1
    print(acc_scores(a, b))
