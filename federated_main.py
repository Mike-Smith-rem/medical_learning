# first define the dataset
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from model.model_combine import UnetCombine, UnetEncoder, UnetDecoder, UnetClassifier, UnetCls, UnetSeg
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

        for name in self.img_names:
            img = Image.open(os.path.join(self.file_dir, 'img', name))
            if name.startswith("0"):
                label = 0
            elif name.startswith("1"):
                label = 1
            elif name.startswith("2"):
                label = 2
            else:
                raise TypeError("not supported class: {}".format(name))
            mask = Image.open(os.path.join(self.file_dir, 'mask', name))

            img = np.array(img)
            label = np.array(label)
            mask = np.array(mask)
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.masks_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            img = self.img_transform(img)
            label = torch.tensor(label)
            mask = self.masks_transform(mask)
            if mask.max() > 1:
                mask = mask / 255
            self.imgs.append(img)
            self.labels.append(label)
            self.masks.append(mask)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx], self.masks[idx]


dataset_train = busi_dataset(train=True)
dataset_test = busi_dataset(train=False)
dataset_test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False)
print("dataset load success!")
print("----dataset_train_length:" + str(len(dataset_train)))
print("----dataset_test_length:" + str(len(dataset_test)))


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, mask = self.dataset[self.idxs[item]]
        return image, label, mask


class LocalUpdate(object):
    def __init__(self, type, args=None, dataset=None, idxs=None):
        self.args = args
        if args is None:
            self.args = {
                "local_bs": 4,
                "lr": 1e-4,
                "momentum": 0.9,
                "local_ep": 10,
            }
        self.type = type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args['local_bs'], shuffle=True)

    def train(self, net):
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
                if not add_unsupervised_train:
                    log_probs = net(images)
                else:
                    log_probs = net(images, self.type)
                if self.type == 'seg':
                    B, C, H, W = log_probs.shape
                    log_probs = log_probs.view(B, C, -1)
                    labels = labels.view(B, -1).long()
                else:
                    B, C = log_probs.shape
                    log_probs = log_probs.view(B, -1)
                    labels = labels.view(-1).long()
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if add_unsupervised_train:
            changed_type = 'cls' if self.type == 'seg' else 'seg'
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])
            for iter in range(self.args['local_ep']):
                batch_loss = []
                for batch_idx, (images, labels, masks) in enumerate(self.ldr_train):
                    if changed_type == 'cls':
                        images, labels = images.to(self.device), labels.to(self.device)
                    elif changed_type == 'seg':
                        images, labels = images.to(self.device), masks.to(self.device)
                    else:
                        raise NotImplementedError("please check the arguments")

                    net.zero_grad()
                    log_probs = net(images, changed_type)
                    if changed_type == 'seg':
                        B, C, H, W = log_probs.shape
                        log_probs = log_probs.view(B, C, -1)
                        # labels = labels.view(B, -1).long()
                    else:
                        B, C = log_probs.shape
                        log_probs = log_probs.view(B, -1)

                    pseudo_values, _ = torch.max(log_probs, dim=1)
                    threshold = 0.7
                    avaliable_index = [i for i in range(B) if pseudo_values[i].min() > threshold]
                    if len(avaliable_index) == 0:
                        print("No pseudo value found")
                        continue
                    preds = torch.index_select(log_probs, dim=0, index=torch.tensor(avaliable_index, dtype=torch.int32))
                    pse = torch.argmax(preds, dim=1)
                    unsupervised_loss = self.loss_func(preds, pse)
                    unsupervised_loss.backward()
                    optimizer.step()
                    batch_loss.append(unsupervised_loss)
                    if batch_idx % 10 == 0:
                        print('Update Epoch Unsupervised_type_{}: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t '
                              'training_set_length {}'.format(
                            changed_type, iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                                100. * batch_idx / len(self.ldr_train), unsupervised_loss.item(),
                            preds.shape[0]))
        return net, sum(epoch_loss) / len(epoch_loss)


# segment the dataset in different patchs

def iid_busi(dataset, num_clients) -> dict:
    """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_clients:
        :return: dict of image index
        """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


clients = ['seg', 'cls', 'cls', 'seg']
# dict_client is {0: [], 1:[], 2:[], 3:[]}
dict_client_train = iid_busi(dataset=dataset_train, num_clients=len(clients))
# dict_client_test = iid_busi(dataset=dataset_test, num_clients=len(clients))
print("client split success!")
print("---------------")
# second set the glob model
# n_channels: channels of image
# n_class1: for cls
# n_class2: for seg
add_unsupervised_train = True
seed = 2024


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    glob_encoder = UnetEncoder(n_channels=3)
    glob_decoder = UnetDecoder(n_class=2)
    glob_classifier = UnetClassifier(n_class=3)

    # ---------
    # third: send the model to other clients
    glob_encoder_parameters = glob_encoder.state_dict()
    glob_decoder_parameters = glob_decoder.state_dict()
    glob_classifier_parameters = glob_classifier.state_dict()

    print("global_model load success!")
    writer = SummaryWriter(log_dir="logs/federated_global" + str(add_unsupervised_train))
    # fourth: train
    glob_epochs = 2
    for iter in range(glob_epochs):
        print("Epoch:" + str(iter))
        # 1. 将参数下发多个客户端
        encoder_parameters = [glob_encoder_parameters for i in range(len(clients))]
        decoder_parameters = [glob_decoder_parameters for i in range(len(clients))]
        cls_parameters = [glob_classifier_parameters for i in range(len(clients))]
        print("split parameters success!")
        # 2. 保存损失
        loss_locals = [0 for i in range(len(clients))]
        loss_globals = []

        # 3. 本地训练，并保存训练后的网络
        for i in range(len(clients)):
            local = LocalUpdate(clients[i], None, dataset_train, dict_client_train)
            if not add_unsupervised_train:
                # encoder part
                local_encoder = UnetEncoder(n_channels=3)
                local_encoder.load_state_dict(encoder_parameters[i])
                # decoder/cls part
                local_decoder_classifier = UnetDecoder(n_class=2) \
                    if clients[i] == 'seg' else UnetClassifier(n_class=3)
                local_decoder_classifier.load_state_dict(decoder_parameters[i]
                                                         if clients[i] == 'seg' else cls_parameters[i])
                # build the final model
                if clients[i] == 'seg':
                    net = UnetSeg(local_encoder, local_decoder_classifier)
                else:
                    net = UnetCls(local_encoder, local_decoder_classifier)

            else:
                net = UnetCombine(n_channels=3, n_class1=3, n_class2=2)
                net.encoder.load_state_dict(encoder_parameters[i])
                net.decoder.load_state_dict(decoder_parameters[i])
                net.classifier.load_state_dict(cls_parameters[i])
            net_parameters, loss = local.train(net=net)
            _encoder_parameters = net_parameters.encoder.state_dict()
            _decoder_parameters = net_parameters.decoder.state_dict() if clients[i] == 'seg' else decoder_parameters[i]
            _cls_parameters = net_parameters.classifier.state_dict() if clients[i] == 'cls' else cls_parameters[i]

            encoder_parameters[i] = copy.deepcopy(_encoder_parameters)
            decoder_parameters[i] = copy.deepcopy(_decoder_parameters) if clients[i] == 'seg' else decoder_parameters[i]
            cls_parameters[i] = copy.deepcopy(_cls_parameters) if clients[i] == 'cls' else cls_parameters[i]
            loss_locals[i] += copy.deepcopy(loss)

        # 4. 聚合网络
        glob_encoder_parameters = FedAvg(encoder_parameters)
        glob_encoder.load_state_dict(glob_encoder_parameters)
        print("encoder aggregated parameters with {} success!".format(len(encoder_parameters)))

        decoders = [decoder_parameters[i] for i in range(len(decoder_parameters)) if clients[i] == 'seg']
        glob_decoder_parameters = FedAvg(decoders)
        glob_decoder.load_state_dict(glob_decoder_parameters)
        print("decoder aggregated parameters with {} success!".format(len(decoder_parameters)))

        classifiers = [cls_parameters[i] for i in range(len(cls_parameters)) if clients[i] == 'cls']
        glob_classifier_parameters = FedAvg(classifiers)
        glob_classifier.load_state_dict(glob_classifier_parameters)
        print("cls aggregated parameters with {} success!".format(len(cls_parameters)))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_globals.append(loss_avg)
        writer.add_scalar('Loss_Global', loss_avg, iter)

        def cls_test(val_loader, model, criterion, device) -> dict:
            model.eval()
            test_dict = {
                'test_loss': 0.,
                'test_acc': 0.,
                # 'test_acc_all': 0.
            }
            with torch.no_grad():
                for img, label, _ in val_loader:
                    img, label = img.to(device), label.to(device)
                    pred = model(img)
                    B, C = pred.shape
                    pred = pred.view(B, -1)
                    label = label.view(-1).long()
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

        cls_model = UnetCls(glob_encoder, glob_classifier)
        acc_avg = cls_test(dataset_test_loader, cls_model, nn.CrossEntropyLoss(), device='cpu')
        acc_avg = acc_avg['test_acc']
        print('Epoch: {}; Test Acc:{}'.format(iter, acc_avg))
        writer.add_scalar('Acc_Global', acc_avg, iter)

        def seg_test(val_loader, model, criterion, device) -> dict:
            model.eval()
            test_dict = {
                'test_loss': 0.,
                'test_dice': 0.,
            }
            with torch.no_grad():
                for img, _, mask in val_loader:
                    img, label = img.to(device), mask.to(device)
                    pred = model(img)
                    B, C, H, W = pred.shape
                    pred = pred.view(B, C, -1)
                    label = label.view(B, -1).long()
                    loss = criterion(pred, label)
                    dice_val = binary_dice_score(label_gt=label, label_pred=pred)
                    test_dict['test_loss'] += loss.item()
                    test_dict['test_dice'] += dice_val['dice']

            test_dict = {
                'test_loss': test_dict['test_loss'] / len(val_loader),
                'test_dice': test_dict['test_dice'] / len(val_loader)
            }
            return test_dict

        seg_model = UnetSeg(glob_encoder, glob_decoder)
        dice_avg = seg_test(dataset_test_loader, seg_model, nn.CrossEntropyLoss(), device='cpu')
        dice_avg = dice_avg['test_dice']
        print('Epoch: {}; Test Dice:{}'.format(iter, dice_avg))
        writer.add_scalar('Dice_Global', dice_avg, iter)


if __name__ == '__main__':
    add_unsupervised_train = True
    main()
    add_unsupervised_train = False
    main()
