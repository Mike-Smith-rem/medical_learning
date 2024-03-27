'''Copyright oyk
Created 10 16:46:22
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_segmentation import Down, DoubleConv, Up


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # 构建隐藏层
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # 前向传播过程
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x


class UNet_Classifier(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Classifier, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 下采样部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.mlp = MLP(1024 // factor, [256], 128)

        # 分类器
        self.classifier = nn.Linear(1024 // factor, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # 应用全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平

        # x = self.mlp(x)
        # 分类器
        x = self.classifier(x)
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    from dataset.busi_dataset import classifer_busi_dataloader
    from torch.utils.data import Dataset
    import os
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    net = UNet_Classifier(n_channels=3, n_classes=3, bilinear=True)


    class BUSICLSDataset(Dataset):
        def __init__(self, path, transform=None):
            self.path = path
            self.names = os.listdir(path)
            self.transform = transform

        def __getitem__(self, idx):
            name = self.names[idx]
            path = os.path.join(self.path, name)
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = torch.tensor(int(name.split('_')[0])).long()
            return img, label

        def __len__(self):
            return len(self.names)


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    busi_train = BUSICLSDataset(path=os.path.join('../dataset/test/train100/all/img'), transform=transform)
    busi_test = BUSICLSDataset(path=os.path.join('../dataset/test/test100/all/img'), transform=transform)

    train_loader = torch.utils.data.DataLoader(busi_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(busi_test, batch_size=32, shuffle=False)

    # criterion = nn.CrossEntropyLoss()
    from train.utils.metric import acc_scores
