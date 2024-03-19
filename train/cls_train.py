import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train.utils.logger import Logging
from train.utils.metric import acc_scores
from .train_utils import load_criterion, load_optimizer, load_scheduler


def cls_train(args: dict, other: dict, model: nn.Module,
              train_loader: DataLoader, val_loader: DataLoader, logger: Logging):
    criterion = load_criterion(args, other)
    optimizer = load_optimizer(args, other, model)
    scheduler = load_scheduler(args, other, optimizer)
    epochs = args['epochs']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = 0.0
    logger.save_file("start training...", epoch=None)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for idx, (img, label) in enumerate(train_loader):
            # img is [B, C, H, W] and label is [B, H, W]
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            pred = model(img)
            pred = F.softmax(pred, dim=1)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            acc = acc_scores(label, pred)['mean_acc'].item()
            epoch_acc += acc

        # save the training progress
        scheduler.step()
        train_dic = {
            'train_loss': epoch_loss / len(train_loader),
            'train_acc': epoch_acc / len(train_loader)
        }
        logger.save_file(train_dic, epoch, prefix="Train")

        # save the testing progress
        test_dict = cls_test(val_loader, model, criterion, device)
        logger.save_file(test_dict, epoch, prefix="Test")

        # save the best model
        test_acc = test_dict['test_acc']
        if test_acc > best_acc:
            best_acc = test_acc
            logger.save_model(model, '_test_acc_' + str(test_acc))


def cls_test(val_loader, model, criterion, device) -> dict:
    model.eval()
    test_dict = {
        'test_loss': 0.,
        'test_acc': 0.,
        # 'test_acc_all': 0.
    }
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            pred = F.softmax(pred, dim=1)
            loss = criterion(pred, label)
            dice_val = acc_scores(label, pred)
            test_dict['test_loss'] += loss.item()
            test_dict['test_acc'] += dice_val['mean_acc'].item()
            # test_dict['test_acc_all'] += dice_val['acc']

    test_dict = {
        'test_loss': test_dict['test_loss'] / len(val_loader),
        'test_acc': test_dict['test_acc'] / len(val_loader),
        # 'test_acc_all': test_dict['test_acc_all'] / len(val_loader)
    }
    return test_dict


if __name__ == '__main__':
    pass
