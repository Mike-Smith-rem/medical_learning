'''Copyright oyk
Created 07 09:08:05
'''
import torch
import os
from torch.utils.data import Dataset
import numpy as np


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, mask = self.dataset[self.idxs[item]]
        return image, label, mask


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

