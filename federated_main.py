# first define the dataset
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from train.utils.metric import acc_scores, binary_dice_score


class busi_dataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.file_dir = os.path.join("dataset/test/train100/all")
        else:
            self.file_dir = os.path.join("dataset/test/test100/all")
        from PIL import Image
        self.img_names = os.listdir(os.path.join(self.file_dir, 'img'))
        self.mask_names = os.listdir(os.path.join(self.file_dir, 'mask'))
        self.imgs = []
        self.labels = []
        self.masks = []

        for idx, name in enumerate(self.img_names):
            img = Image.open(os.path.join(self.file_dir, 'img', name))\
                .resize((256, 256))
            # img_0 = transforms.Resize((256, 256))(img)
            # img_0.save('origin_image_0.png')
            if name.startswith("0"):
                label = 0
            elif name.startswith("1"):
                label = 1
            elif name.startswith("2"):
                label = 2
                continue
            else:
                raise TypeError("not supported class: {}".format(name))
            mask = Image.open(os.path.join(self.file_dir, 'mask', self.mask_names[idx]))\
                .resize((256, 256)).convert('L')
            # import matplotlib.pyplot as plt
            # if name.startswith("0_13"):
            #     print(name)
            #     plt.imshow(mask)
            #     plt.show()

            img = np.array(img).astype(np.uint8) / 255
            label = np.array(label)
            mask = np.array(mask).astype(np.uint8) / 255
            # print(mask.max(), mask.min())
            # self.img_transform = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((256, 256)),
            #     transforms.ToTensor(),
            #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])
            # self.masks_transform = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Grayscale(),
            #     transforms.Resize((256, 256)),
            #     transforms.ToTensor()
            # ])
            # try:
            #     img = self.img_transform(img)
            #     img_show = transforms.ToPILImage().__call__(img)
            #     img_show = Image.fromarray(img_show)
            #     img_show.save('origin_image_l.png')
            # except:
            #     print(name)
            # label = torch.tensor(label)
            # mask = self.masks_transform(mask)
            # mask = mask.int()
            # print(mask.max())
            # if mask.max() > 1:
            #     mask = mask // 255
            # print(mask.sum())
            # print(mask.shape)
            # mask = np.array(mask).squeeze(axis=0).astype(np.uint8) * 255
            # print(mask.shape)
            # # import matplotlib.pyplot as plt
            # image = Image.fromarray(mask)
            # # 显示图像
            # # plt.imshow(image, cmap='gray')
            # image.save("gray_image.jpg")
            # import sys
            # sys.exit(1)

            self.imgs.append(img)
            self.labels.append(label)
            self.masks.append(mask)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return torch.tensor(self.imgs[idx]).permute(2, 0, 1), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.masks[idx])


threshold = 0.8
dataset_train = busi_dataset(train=True)
dataset_test = busi_dataset(train=False)
dataset_train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
dataset_test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, mask = self.dataset[self.idxs[item]]
        return image, label, mask


def iid_busi(dataset, num_clients) -> dict:
    """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_clients:
        :return: dict of image index
    """
    # dataset comprised of 0, 1, 2
    # special handle
    num_class = 3
    class0_items = int(len(dataset) / num_clients / num_class)
    dict_clients, class0_all_idxs = {}, [i for i in range(len(dataset) // num_class)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(class0_all_idxs, class0_items, replace=False))
        class0_all_idxs = list(set(class0_all_idxs) - dict_clients[i])
    for i in range(num_clients):
        class0_index = list(dict_clients[i])
        class_all_index = list(dict_clients[i])
        for j in class0_index:
            t = num_class
            for k in range(1, t):
                class_all_index.append(j + len(dataset) // num_class * k)
        dict_clients[i] = set(class_all_index)

    # print the answer
    # for i in range(len(dict_clients)):
    #     d = dict_clients[i]
    #     print(sorted(d))
    return dict_clients


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


clients = ['cls', 'seg', 'cls', 'seg', 'seg', 'cls']
seed = 2024
dict_client_train = iid_busi(dataset_train, len(clients))


def single_test(model, test_loader, device, args, log, iter, type, last_round=False):
    model.to(device)
    model.eval()

    def seg_test(val_loader, model, criterion, device) -> dict:
        model.to(device)
        model.eval()
        test_dict = {
            'test_loss': 0.,
            'test_dice': 0.,
        }
        with torch.no_grad():
            time = 0
            for img, _, mask in val_loader:
                img, label = img.to(device), mask.to(device)
                pred = model(img, 'seg')
                pred = F.softmax(pred, dim=1)
                B, C, H, W = pred.shape
                if last_round:
                    # pred = torch.argmax(pred, dim=1)
                    pred = (pred > 0.5).int()
                    pred = pred.detach().cpu().squeeze(1).numpy()
                    label = label.detach().cpu().squeeze(1).numpy()
                    for i in range(B):
                        img = (pred[i]) * 255
                        lb = (label[i]) * 255
                        log.add_image("Test Image", img, global_step=time, dataformats='HW')
                        log.add_image('Test Mask', lb, global_step=time, dataformats='HW')
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

    def cls_test(val_loader, model, criterion, device) -> dict:
        model.to(device)
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
                img, label = img.to(device), label.to(device)
                pred = model(img, 'cls')
                B, C = pred.shape
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

        test_dict = {
            'test_loss': test_dict['test_loss'],
            'test_acc': test_dict['test_acc'],
        }
        return test_dict

    if args['type'] == 'cls':
        acc = cls_test(test_loader, model, nn.CrossEntropyLoss(), device)
        if type == 'train':
            log.add_scalar("Cls Train Acc", acc['test_acc'], iter)
            log.add_scalar('Cls Train Loss', acc['test_loss'], iter)
            print("Cls Train Acc: {}".format(acc))
        else:
            log.add_scalar("Cls Test Acc", acc['test_acc'], iter)
            log.add_scalar('Cls Test Loss', acc['test_loss'], iter)
            print("Cls Test Acc: {}".format(acc))
        return acc['test_loss'], acc['test_acc']
    else:
        from train.utils.loss import BinaryDiceLoss
        criterion = BinaryDiceLoss()
        dice = seg_test(test_loader, model, criterion, device)
        if type == 'train':
            log.add_scalar("Seg Train Dice", dice['test_dice'], iter)
            log.add_scalar('Seg Train Loss', dice['test_loss'], iter)
            print("Seg Train Dice: {}".format(dice))
        else:
            log.add_scalar("Seg Test Dice", dice['test_dice'], iter)
            log.add_scalar('Seg Test Loss', dice['test_loss'], iter)
            print("Seg Test Dice: {}".format(dice))
        return dice['test_loss'], dice['test_dice']


def main():
    from model.unet_clear.unet import Unet
    model = Unet(num_classes=2, pretrained = False)
    glob_model_states = model.state_dict()

    writer = SummaryWriter(log_dir="logs/federated_global")
    # fourth: train
    glob_epochs = 200
    for iter in range(glob_epochs):
        print("Epoch:" + str(iter))
        parameters = [glob_model_states for _ in range(len(clients))]
        loss_locals = [0 for _ in range(len(clients))]
        loss_globals = []

        # 3. 本地训练，并保存训练后的网络
        for i in range(len(clients)):
            from federated.clients import LocalUpdate
            local = LocalUpdate(clients[i], None, dataset_train, dict_client_train[i],
                                multi_task=True, log=SummaryWriter(f'{clients[i]}_{i}'))

            net = Unet(num_classes=2, pretrained=False)
            net.load_state_dict(parameters[i])

            net_parameters, loss = local.train(net=net)
            net_parameters = net_parameters.state_dict()

            parameters[i] = copy.deepcopy(net_parameters)
            loss_locals[i] += copy.deepcopy(loss)

        glob_model_states = FedAvg(parameters)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_globals.append(loss_avg)
        writer.add_scalar('Loss_Global', loss_avg, iter)

        model.load_state_dict(glob_model_states)
        # save memory
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        single_test(model=model, test_loader=dataset_train_loader,
                    device=device, args={'type': 'cls'}, log=writer, iter=iter, type='train')

        # device = torch.device('cpu')
        single_test(model=model, test_loader=dataset_train_loader,
                    device=device, args={'type': 'seg'}, log=writer, iter=iter, type='train')

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        single_test(model=model, test_loader=dataset_test_loader,
                    device=device, args={'type': 'cls'}, log=writer, iter=iter, type='test')

        # device = torch.device('cpu')
        single_test(model=model, test_loader=dataset_test_loader,
                    device=device, args={'type': 'seg'}, log=writer, iter=iter, type='test')

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def single_train(model, train_set, idx, test_loader, device, args_, log):
#     model.to(device)
#     model.train()
#     args = args_
#
#     criterion = nn.CrossEntropyLoss()
#     # from train.utils.loss import BinaryDiceLoss
#     # criterion = BinaryDiceLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
#     # optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
#     # train_loader = DataLoader(DatasetSplit(train_set, idx), batch_size=args['bs'], shuffle=True)
#     # 假设全部训练
#     train_loader = DataLoader(train_set, batch_size=args['bs'], shuffle=True)
#     epoch_loss = []
#     epoch_acc = []
#     epoch_test_loss = []
#     epoch_test_acc = []
#     for epoch in range(args['epochs']):
#         batch_loss = []
#         for imgs, labels, masks in tqdm(train_loader):
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#             masks = masks.to(device)
#             optimizer.zero_grad()
#             log_probs = model(imgs)
#             # log_probs = model(imgs, args['type'])
#             log_probs = F.softmax(log_probs, dim=1)
#             # print(imgs.dtype)
#             # log_probs = F.softmax(log_probs, dim=1)
#             # if args['type'] == 'seg':
#             #     B, C, H, W = log_probs.shape
#             #     log_probs = log_probs.view(B, C, -1)
#             #     labels = masks.view(B, -1).long()
#             # else:
#             #     B, C = log_probs.shape
#             #     log_probs = log_probs.view(B, -1)
#             #     labels = labels.view(-1).long()
#             # print(log_probs.shape, labels.shape)
#             if args['type'] == 'seg':
#                 # if masks.shape[1] == 1:
#                 #     masks = masks.squeeze(dim=1).long()
#                 loss = criterion(log_probs, masks)
#             else:
#                 labels = labels.long()
#                 loss = criterion(log_probs, labels)
#             loss.backward()
#             optimizer.step()
#             batch_loss.append(loss.item())
#         if log is not None:
#             single_test(model, train_loader, device, args, log, epoch, type='train')
#             single_test(model, test_loader, device, args, log, epoch, type='test')
#             epoch_loss.append(sum(batch_loss) / len(batch_loss))
#             log.add_scalar('Loss_Single', epoch_loss[-1], epoch)
#         else:
#             train_loss, train_acc = single_test(model, train_loader, device, args, log, epoch, type='train')
#             test_loss, test_acc = single_test(model, test_loader, device, args, log, epoch, type='test')
#             epoch_loss.append(train_loss)
#             epoch_acc.append(train_acc)
#             epoch_test_loss.append(test_loss)
#             epoch_test_acc.append(test_acc)
#
#         print('Epoch: {}; Loss: {}'.format(epoch, epoch_loss[-1]))
#
#     if log is None:
#         plot(epoch_loss)
#         plot(epoch_acc, label='Training Acc', title='Training Acc', y_label='Training Acc')
#         plot(epoch_test_loss, label='Testing Loss', title='Testing Loss', y_label='Testing Loss')
#         plot(epoch_acc, label='Testing Acc', title='Testing Acc', y_label='Testing Acc')


# def plot(values, label='Loss', title='Training Loss', y_label='Loss'):
#     import matplotlib.pyplot as plt
#
#     steps = list(range(1, len(values) + 1))
#
#     # 绘制损失函数曲线图
#     plt.plot(steps, values, label=label)
#     plt.title(title)
#     plt.xlabel('Steps')
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.grid(True)
#
#     # 保存图像到文件
#     plt.savefig(title + '_plot.png')
#     plt.clf()


# def single_test(model, test_loader, device, args, log, iter, type, last_round=False):
#     model.to(device)
#     model.eval()
#
#     def seg_test(val_loader, model, criterion, device) -> dict:
#         model.to(device)
#         model.eval()
#         test_dict = {
#             'test_loss': 0.,
#             'test_dice': 0.,
#         }
#         with torch.no_grad():
#             time = 0
#             for img, _, mask in val_loader:
#                 img, label = img.to(device), mask.to(device)
#                 pred = model(img, 'seg')
#                 # pred = model(img)
#                 pred = F.sigmoid(pred)
#                 B, C, H, W = pred.shape
#                 if last_round:
#                     # pred = torch.argmax(pred, dim=1)
#                     pred = (pred > 0.5).int()
#                     pred = pred.detach().cpu().squeeze(1).numpy()
#                     label = label.detach().cpu().squeeze(1).numpy()
#                     for i in range(B):
#                         img = (pred[i]) * 255
#                         lb = (label[i]) * 255
#                         log.add_image("Test Image", img, global_step=time, dataformats='HW')
#                         log.add_image('Test Mask', lb, global_step=time, dataformats='HW')
#                         time = time + 1
#                     continue
#                 # if label.shape[1] == 1:
#                 #     label = label.squeeze(1)
#                 # label = label.long()
#                 loss = criterion(pred, label)
#                 dice_val = binary_dice_score(label_gt=label, label_pred=pred)
#                 # from train.utils.metric import f_score
#                 # dice_val = f_score(pred, label)
#                 test_dict['test_loss'] += loss.item()
#                 test_dict['test_dice'] += dice_val['dice'].item()
#                 # test_dict['test_dice'] += dice_val.item()
#
#         test_dict = {
#             'test_loss': test_dict['test_loss'] / len(val_loader),
#             'test_dice': test_dict['test_dice'] / len(val_loader)
#         }
#         return test_dict
#
#     def cls_test(val_loader, model, criterion, device) -> dict:
#         model.to(device)
#         model.eval()
#         test_dict = {
#             'test_loss': 0.,
#             'test_acc': 0.,
#             # 'test_acc_all': 0.
#         }
#         with torch.no_grad():
#             preds = []
#             labels = []
#             loss = 0
#             for img, label, _ in val_loader:
#                 img, label = img.to(device), label.to(device)
#                 # pred = model(img, 'cls')
#                 pred = model(img)
#                 B, C = pred.shape
#                 # load pred and labels
#                 pred = pred.view(B, -1)
#                 label = label.view(-1).long()
#                 preds.append(pred)
#                 labels.append(label)
#                 loss += criterion(pred, label).item()
#             label = torch.cat(labels, dim=0)
#             pred = torch.cat(preds, dim=0)
#             dice_val = acc_scores(label, pred)
#             test_dict['test_loss'] = loss
#             test_dict['test_acc'] = dice_val['mean_acc'].item()
#             # test_dict['test_acc_all'] += dice_val['acc']
#
#         test_dict = {
#             'test_loss': test_dict['test_loss'],
#             'test_acc': test_dict['test_acc'],
#             # 'test_acc_all': test_dict['test_acc_all'] / len(val_loader)
#         }
#         return test_dict
#
#     if args['type'] == 'cls':
#         acc = cls_test(test_loader, model, nn.CrossEntropyLoss(), # 'cpu')
#                        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#         if type == 'train':
#             log.add_scalar("Train Acc", acc['test_acc'], iter)
#             log.add_scalar('Train Loss', acc['test_loss'], iter)
#             print("Train Acc Single: {}".format(acc))
#         else:
#             log.add_scalar("Test Acc", acc['test_acc'], iter)
#             log.add_scalar('Test Loss', acc['test_loss'], iter)
#             print("Test Acc Single: {}".format(acc))
#         return acc['test_loss'], acc['test_acc']
#     else:
#         from train.utils.loss import BinaryDiceLoss
#         criterion = BinaryDiceLoss()
#         dice = seg_test(test_loader, model, criterion, # 'cpu')
#                         torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
#         if type == 'train':
#             log.add_scalar("Train Dice", dice['test_dice'], iter)
#             log.add_scalar('Train Loss', dice['test_loss'], iter)
#             print("Train Dice Single: {}".format(dice))
#         else:
#             log.add_scalar("Test Dice", dice['test_dice'], iter)
#             log.add_scalar('Test Loss', dice['test_loss'], iter)
#             print("Test Dice Single: {}".format(dice))
#         return dice['test_loss'], dice['test_dice']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()
    # args = {
    #     "bs": 8,
    #     "lr": 0.1,
    #     "momentum": 0.1,
    #     "epochs": 200,
    #     "type": 'cls'
    # }