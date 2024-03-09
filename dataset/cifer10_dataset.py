'''Copyright oyk
Created 28 22:01:25
'''
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

import cv2
import numpy as np
import pickle


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dicts = pickle.load(fo, encoding='latin1')
    fo.close()
    return dicts


def cifar10_to_images():
    tar_dir = os.path.join('cifar-10-batches-py')  # 原始数据库目录
    train_root_dir = '../data/cifar10/train/'  # 图片保存目录
    test_root_dir = '../data/cifar10/test/'
    if not os.path.exists(train_root_dir):
        os.makedirs(train_root_dir)
    if not os.path.exists(test_root_dir):
        os.makedirs(test_root_dir)
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    label_names = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]
    for j in range(1, 6):
        dataName = os.path.join(tar_dir, "data_batch_" + str(j))
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")

        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = train_root_dir + str(Xtr['labels'][i]) + '_' + label_names[Xtr['labels'][i]] + '_' + str(
                i + (j - 1) * 10000) + '.jpg'  # label+class+index
            cv2.imwrite(picName, img)
        print(dataName + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    testXtr = unpickle(tar_dir + "/test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = test_root_dir + str(testXtr['labels'][i]) + '_' + label_names[testXtr['labels'][i]] + '_' + str(
            i) + '.jpg'
        cv2.imwrite(picName, img)
    print("test_batch loaded.")


def classifier_cifer10_dataset(root=os.getcwd(), bs=32):
    transform = transforms.Compose(
        # B,G,R 三个通道归一化 标准差为 0.5， 方差为0.5
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将父目录添加到Python路径中
    sys.path.append(current_dir)
    root = os.path.join(current_dir)

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == '__main__':
    # cifar10_to_images()
    path = os.path.join('', 'cifar-10-batches-py')
    if os.path.exists(path):
        print(True)
    else:
        print(False)
    train_loader, test_loader = classifier_cifer10_dataset()

    print(len(train_loader))
    print(len(test_loader))
