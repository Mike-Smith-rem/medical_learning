'''Copyright oyk
Created 11 17:27:43
'''

import os

# import cv2
# use pli instead
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


def delete_ipynb(path):
    # 指定要删除的文件名
    import shutil

    # 指定要删除的文件夹路径
    folder_path = os.path.join(path, ".ipynb_checkpoints")

    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 如果存在，则递归删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        # 如果文件夹不存在，打印消息
        print(f"Folder '{folder_path}' does not exist.")


class BUSI_classifier_dataSet(Dataset):
    # benign is 0, malignant is 1, normal is 2
    def __init__(self, path, label, transform=None, target_transform=None):
        self.img_dir = path
        self.img_list = os.listdir(path)
        # delete_ipynb(self.img_dir)
        self.label_list = [label for i in range(len(self.img_list))]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_path = os.path.join(self.img_dir, img_name)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.open(img_path)
        img = np.array(img)
        label = self.label_list[item]
        if self.transform:
            print(img.shape)
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class BUSI_segmentation_dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.file_path = path
        self.img_path = os.path.join(path, 'img')
        self.mask_path = os.path.join(path, 'mask')
        # delete_ipynb(self.img_path)
        # delete_ipynb(self.mask_path)
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        mask_name = self.mask_list[item]
        img_path = os.path.join(self.img_path, img_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("test.png", img)
        # import sys
        # sys.exit(1)
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(img_path))
        # mask = cv2.imread(mask_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        if mask.max() > 1:
            mask = mask / 255

        return img, mask


def segment_busi_dataLoader(load_normal_data=True, bs=32, info=None):
    # normMean = [0.32748958, 0.32748246, 0.3274379]
    # normStd = [0.22094825, 0.2209481, 0.22093095]
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((256, 256))
    ])
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将父目录添加到Python路径中
    sys.path.append(current_dir)
    if load_normal_data:
        train_path = os.path.join(current_dir, "classifer/train100/all")
        test_path = os.path.join(current_dir, "classifer/test100/all")
    else:
        train_path = os.path.join(current_dir, "segment_no_normal/train")
        test_path = os.path.join(current_dir, "segment_no_normal/test")
    train_dataset = BUSI_segmentation_dataset(train_path, img_transform, target_transform)
    test_dataset = BUSI_segmentation_dataset(test_path, img_transform, target_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
    return train_dataloader, test_dataloader


def classifer_busi_dataloader(load_normal_data=True, bs=32, info=None, debug=True):
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将父目录添加到Python路径中
    sys.path.append(current_dir)
    path0 = os.path.join(current_dir, "test/train100/benign")
    path1 = os.path.join(current_dir, "test/train100/malignant")
    path2 = os.path.join(current_dir, "test/train100/normal")

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset0 = BUSI_classifier_dataSet(path0, 0, img_transform)
    dataset1 = BUSI_classifier_dataSet(path1, 1, img_transform)
    dataset2 = BUSI_classifier_dataSet(path2, 2, img_transform)

    path0_test = os.path.join(current_dir, "test/test100/benign")
    dataset0_test = BUSI_classifier_dataSet(path0_test, 0, img_transform)
    path1_test = os.path.join(current_dir, "test/test100/malignant")
    dataset1_test = BUSI_classifier_dataSet(path1_test, 1, img_transform)
    path2_test = os.path.join(current_dir, "test/test100/normal")
    dataset2_test = BUSI_classifier_dataSet(path2_test, 2, img_transform)
    if load_normal_data:
        train_concat_dataset = ConcatDataset([dataset0, dataset1, dataset2])
        test_concat_dataset = ConcatDataset([dataset0_test, dataset1_test, dataset2_test])
    else:
        train_concat_dataset = ConcatDataset([dataset0, dataset1])
        test_concat_dataset = ConcatDataset([dataset0_test, dataset1_test])

    # 创建 DataLoader 对象，加载训练集和测试集
    train_dataloader = DataLoader(train_concat_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_concat_dataset, batch_size=bs, shuffle=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    path = os.path.join("segment/train")
    normMean = [0.32748958, 0.32748246, 0.3274379]
    normStd = [0.22094825, 0.2209481, 0.22093095]
    img_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        # transforms.Normalize(normMean, normStd)
    ])
    train_dataloader, test_dataloader = segment_busi_dataLoader(bs=1)
    for img, mask in train_dataloader:
        print(img.shape)
        print(mask.shape)
        # img = img.reshape(img.shape[1], img.shape[2], img.shape[3])
        # mask = mask.reshape(mask.shape[1], mask.shape[2], mask.shape[3])
        # plt.imshow(img[0], cmap='gray')
        # plt.show()

        break
    # from PIL import Image
    # import numpy as np
    #
    # img = Image.open(os.path.join('classifer/benign/img/0.png'))
    # print(type(img))
    # print(img)
    # img_np = np.array(img)
    # print(img_np.shape)
    # img_tensor = torch.Tensor(img_np)

    # img = Image.fromarray(img_np).save(os.path.join("new_image.png"))

