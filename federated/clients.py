'''Copyright oyk
Created 28 15:08:56
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from .dataset.dataset import DatasetSplit

import sys
sys.path.append("..")
from train.utils.loss import BinaryDiceLoss
from train.utils.metric import binary_dice_score, acc_scores
from tqdm import tqdm


class LocalUpdate(object):
    def __init__(self, type, args=None, dataset=None, idxs=None, multi_task=False, log=None):
        self.args = args
        if args is None:
            self.args = {
                "local_bs": 8,
                "lr": 1e-2,
                "momentum": 0,
                "local_ep": 10,
            }
        self.type = type
        self.multi_task = multi_task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type == 'cls':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = BinaryDiceLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args['local_bs'], shuffle=True)
        self.log = log

    def train(self, net):
        net.to(self.device)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])

        epoch_loss = []
        for iter in range(self.args['local_ep']):
            batch_loss = []
            for batch_idx, (images, labels, masks) in enumerate(self.ldr_train):
                if self.type == 'seg':
                    images, labels = images.to(self.device), masks.to(self.device)
                elif self.type == 'cls':
                    images, labels = images.to(self.device), labels.to(self.device)
                else:
                    raise NotImplementedError("please check the arguments")

                net.zero_grad()
                if not self.multi_task:
                    log_probs = net(images)
                else:
                    log_probs = net(images, self.type)
                # if self.type == 'seg':
                #     B, C, H, W = log_probs.shape
                #     log_probs = log_probs.view(B, C, -1)
                #     labels = labels.view(B, -1).long()
                # else:
                #     B, C = log_probs.shape[0], log_probs.shape[1]
                #     log_probs = log_probs.view(B, -1)
                #     labels = labels.view(-1).long()
                labels = labels.long()
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train),
                #               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # if add_unsupervised_train:
        #     changed_type = 'cls' if self.type == 'seg' else 'seg'
        #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])
        #     for iter in range(self.args['local_ep']):
        #         batch_loss = []
        #         for batch_idx, (images, labels, masks) in enumerate(self.ldr_train):
        #             if changed_type == 'cls':
        #                 images, labels = images.to(self.device), labels.to(self.device)
        #             elif changed_type == 'seg':
        #                 # print(mask.shape)
        #                 images, labels = images.to(self.device), masks.to(self.device)
        #                 # print(images.shape, labels.shape)
        #                 # print(mask.shape)
        #             else:
        #                 raise NotImplementedError("please check the arguments")
        #
        #             net.zero_grad()
        #             log_probs = net(images, changed_type)
        #             if changed_type == 'seg':
        #                 B, C, H, W = log_probs.shape
        #                 log_probs = log_probs.view(B, C, -1)
        #                 # labels = labels.view(B, -1).long()
        #             else:
        #                 B, C = log_probs.shape
        #                 log_probs = log_probs.view(B, -1)
        #
        #             pseudo_values, _ = torch.max(log_probs, dim=1)
        #             avaliable_index = [i for i in range(B) if pseudo_values[i].min() > threshold]
        #             if len(avaliable_index) == 0:
        #                 # print("No pseudo value found")
        #                 continue
        #             # print(mask.shape)
        #             # print(images.shape)
        #             # print(labels.shape)
        #             preds = torch.index_select(log_probs, dim=0, index=torch.tensor(avaliable_index, dtype=torch.long).
        #                                        to(self.device))
        #             pse = torch.argmax(preds, dim=1)
        #             unsupervised_loss = self.loss_func(preds, pse)
        #             unsupervised_loss.backward()
        #             optimizer.step()
        #             batch_loss.append(unsupervised_loss)
        #             if batch_idx % 10 == 0:
        #                 print('Update Epoch Unsupervised_type_{}: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t '
        #                       .format(changed_type, iter, batch_idx * len(images), len(self.ldr_train),
        #                               100. * batch_idx / len(self.ldr_train), unsupervised_loss.item()))
        return net, sum(epoch_loss) / len(epoch_loss)

    def test(self, net, test_loader, test_type, iter, last_round=False):
        net.to(self.device)
        net.eval()

        def seg_test(val_loader, model, criterion) -> dict:
            model.to(self.device)
            model.eval()
            test_dict = {
                'test_loss': 0.,
                'test_dice': 0.,
            }
            with torch.no_grad():
                time = 0
                for img, _, mask in val_loader:
                    img, label = img.to(self.device), mask.to(self.device)
                    pred = model(img, 'seg') if self.multi_task else model(img)
                    pred = F.sigmoid(pred)
                    B, C, H, W = pred.shape
                    if last_round:
                        # pred = torch.argmax(pred, dim=1)
                        pred = (pred > 0.5).int()
                        pred = pred.detach().cpu().squeeze(1).numpy()
                        label = label.detach().cpu().squeeze(1).numpy()
                        for i in range(B):
                            img = (pred[i]) * 255
                            lb = (label[i]) * 255
                            if self.log is not None:
                                self.log.add_image("Test Image", img, global_step=time, dataformats='HW')
                                self.log.add_image('Test Mask', lb, global_step=time, dataformats='HW')
                            time = time + 1
                        continue
                    loss = criterion(pred, label)
                    dice_val = binary_dice_score(label_gt=label, label_pred=pred)
                    test_dict['test_loss'] += loss.item()
                    test_dict['test_dice'] += dice_val['dice'].item()

            test_dict = {
                'test_loss': test_dict['test_loss'] / len(val_loader),
                'test_dice': test_dict['test_dice'] / len(val_loader)
            }
            return test_dict

        def cls_test(val_loader, model, criterion) -> dict:
            model.to(self.device)
            model.eval()
            test_dict = {
                'test_loss': 0.,
                'test_acc': 0.,
                # 'test_acc_all': 0.
            }
            with torch.no_grad():
                preds = []
                labels = []
                loss = 0
                for img, label, _ in val_loader:
                    img, label = img.to(self.device), label.to(self.device)
                    # pred = model(img, 'cls')
                    pred = model(img)
                    B, C = pred.shape
                    # load pred and labels
                    pred = pred.view(B, -1)
                    label = label.view(-1).long()
                    preds.append(pred)
                    labels.append(label)
                    loss += criterion(pred, label).item()
                label = torch.cat(labels, dim=0)
                pred = torch.cat(preds, dim=0)
                dice_val = acc_scores(label, pred)
                test_dict['test_loss'] = loss
                test_dict['test_acc'] = dice_val['mean_acc'].item()
                # test_dict['test_acc_all'] += dice_val['acc']

            test_dict = {
                'test_loss': test_dict['test_loss'],
                'test_acc': test_dict['test_acc'],
                # 'test_acc_all': test_dict['test_acc_all'] / len(val_loader)
            }
            return test_dict

        if self.type == 'cls':
            acc = cls_test(test_loader, net, nn.CrossEntropyLoss())
            if test_type == 'train':
                self.log.add_scalar("Train Acc", acc['test_acc'], iter)
                self.log.add_scalar('Train Loss', acc['test_loss'], iter)
                print("Train Acc Single: {}".format(acc))
            else:
                self.log.add_scalar("Test Acc", acc['test_acc'], iter)
                self.log.add_scalar('Test Loss', acc['test_loss'], iter)
                print("Test Acc Single: {}".format(acc))
            return acc['test_loss'], acc['test_acc']
        else:
            from ..train.utils.loss import BinaryDiceLoss
            criterion = BinaryDiceLoss()
            dice = seg_test(test_loader, net, criterion)
            if test_type == 'train':
                self.log.add_scalar("Train Dice", dice['test_dice'], iter)
                self.log.add_scalar('Train Loss', dice['test_loss'], iter)
                print("Train Dice Single: {}".format(dice))
            else:
                self.log.add_scalar("Test Dice", dice['test_dice'], iter)
                self.log.add_scalar('Test Loss', dice['test_loss'], iter)
                print("Test Dice Single: {}".format(dice))
            return dice['test_loss'], dice['test_dice']
