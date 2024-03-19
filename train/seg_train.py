import torch
from torch import nn
from torch.utils.data import DataLoader

from .train_utils import load_criterion, load_optimizer, load_scheduler
from train.utils.logger import Logging
from train.utils.metric import binary_dice_score


def seg_train(args: dict, other: dict, model: nn.Module,
              train_loader: DataLoader, val_loader: DataLoader, logger: Logging):
    criterion = load_criterion(args, other)
    optimizer = load_optimizer(args, other, model)
    scheduler = load_scheduler(args, other, optimizer)
    epochs = args['epochs']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_dice = 0.0
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        for idx, (img, label) in enumerate(train_loader):
            # img is [B, C, H, W] and label is [B, H, W]
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            pred = model(img)

            loss = criterion(pred, label.squeeze(1).long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            dice = binary_dice_score(label_gt=label, label_pred=pred)['dice'].item()
            epoch_dice += dice

        scheduler.step()
        # save the training progress
        train_dic = {
            'train_loss': epoch_loss / len(train_loader),
            'train_dice': epoch_dice / len(train_loader)
        }
        logger.save_file(train_dic, epoch, prefix="Train")

        # save the testing progress
        test_dict = seg_test(val_loader, model, criterion, device)
        logger.save_file(test_dict, epoch, prefix="Test")

        # save the best model
        test_dice = test_dict['test_dice']
        if test_dice > best_dice:
            best_dice = test_dice
            logger.save_model(model, '_test_dice_' + str(test_dice))


def seg_test(val_loader, model, criterion, device) -> dict:
    model.eval()
    test_dict = {
        'test_loss': 0.,
        'test_dice': 0.,
    }
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label.squeeze(1).long())
            dice_val = binary_dice_score(label_gt=label, label_pred=pred)
            test_dict['test_loss'] += loss.item()
            test_dict['test_dice'] += dice_val['dice'].item()

    test_dict = {
        'test_loss': test_dict['test_loss'] / len(val_loader),
        'test_dice': test_dict['test_dice'] / len(val_loader)
    }
    return test_dict


if __name__ == '__main__':
    pass
