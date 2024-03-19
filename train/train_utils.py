import torch
from torch import nn
from utils.loss import FocalLoss, SoftDiceLoss


def load_criterion(args: dict, other: dict) -> nn.Module:
    if args['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args['criterion'] == 'FocalLoss':
        criterion = FocalLoss()
        if other is not None and other['alpha'] is not None and other['num_classes'] is not None:
            criterion = FocalLoss(alpha=other['alpha'], num_classes=other['num_classes'])
    elif args['criterion'] == 'SoftDiceLoss':
        criterion = SoftDiceLoss()
        if other is not None and other['alpha'] is not None and other['num_classes'] is not None:
            criterion = SoftDiceLoss(alpha=other['alpha'], num_classes=other['num_classes'])
    else:
        raise NotImplementedError("the loss {} not implemented".format(args['criterion']))
    return criterion


def load_optimizer(args: dict, other: dict, model: nn.Module) -> torch.optim.Optimizer:
    if args['optimizer'] == 'sgd':
        assert other['momentum'] is not None and other['weight_decay'] is not None
        optimizer = torch.optim.SGD(model.parameters(), lr=other['lr'],
                                    momentum=other['momentum'], weight_decay=other['weight_decay'])
    elif args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=other['lr'])
    else:
        raise NotImplementedError("the optimizer {} not implemented".format(args['optimizer']))
    return optimizer


def load_scheduler(args: dict, other: dict, optimizer: torch.optim) -> torch.optim.lr_scheduler:
    if args['lr_scheduler'] == 'LambdaLR':
        assert other['lr'] is not None and other['lr_lambda'] is not None
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=other['lr_lambda'])
    elif args['lr_scheduler'] == 'StepLR':
        assert other['lr'] is not None and other['step_size'] is not None and other['lr_gamma'] is not None
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=other['step_size'], gamma=other['lr_gamma'])
    else:
        raise NotImplementedError("the lr_scheduler {} not implemented".format(args['lr_scheduler']))
    return scheduler
