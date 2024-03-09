import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils.logger import Logging
from .utils.metric import acc_scores
from .train_utils import load_criterion, load_optimizer, load_scheduler


def cls_train(args: dict, other: dict, model: nn.Module,
              train_loader: DataLoader, val_loader: DataLoader, logger: Logging):
    criterion = load_criterion(args, other)
    optimizer = load_optimizer(args, other, model)
    scheduler = load_scheduler(args, other, optimizer)
    epochs = args['epochs']
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)

    best_acc = 0.0
    logger.save_file("start training...", epoch=None)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for idx, (img, label) in enumerate(train_loader):
            # img is [B, C, H, W] and label is [B, H, W]
            # print(img.shape, label.shape)
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            pred = model(img)
            # print(pred.shape)
            # print(label.shape)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # save the training progress
        scheduler.step()

        epoch_acc = cls_test(train_loader, model, criterion, device)
        train_dic = {
            'train_loss': epoch_loss / len(train_loader),
            'train_acc': epoch_acc['test_acc']
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
    test_dict = {}

    with torch.no_grad():
        correct = 0
        total = 0
        test_dict['test_loss'] = 0.
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label)

            pred = torch.argmax(pred, dim=1)
            correct += (label == pred).int().sum().item()
            total += label.size(0)

            test_dict['test_loss'] += loss.item()
        test_dict['test_acc'] = correct / total

    return test_dict


if __name__ == '__main__':
    pass
