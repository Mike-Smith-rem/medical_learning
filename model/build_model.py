import torch
from torch import nn
from torchvision.models import resnet18, vgg16

from unet.model_classifier import UNet_Classifier
from unet.model_segmentation import freeze_encoder
from unet.model_segmentation import UNet


def build_cls_model(model_name, config, mode=None, mode_path=None):
    n_channel = config['n_channel']
    n_class = config['n_class']
    if model_name == 'unet_classifier':
        model = UNet_Classifier(n_channel, n_class)
        if mode is not None:
            _model = UNet(n_channels=3, n_classes=2)
            _model.load_state_dict(torch.load(mode_path))
            freeze_encoder(model, _model)

    elif model_name == 'resnet18':
        # resnet18 channel must be 3
        model = resnet18(pretrained=True)
        features = model.fc.in_features
        model.fc = nn.Linear(features, n_class)

    elif model_name == 'vgg16':
        model = vgg16(pretrained=True)
        features = model.fc.in_features
        model.fc = nn.Linear(features, n_class)
        model.fc = nn.Sequential(
            nn.Linear(features, n_class),
            nn.Softmax(dim=1),
        )
    else:
        raise NotImplementedError(f'Unsupported model: {model_name}')
    return model


def build_seg_model(model_name, config, mode=None, mode_path=None):
    n_channel = config['n_channel']
    n_class = config['n_class']
    if model_name == 'unet':
        model = UNet(n_channels=n_channel, n_classes=n_class)
        if mode is not None:
            _model = UNet_Classifier(n_channels=3, n_classes=3)
            _model.load_state_dict(torch.load(mode_path))
            freeze_encoder(model, _model)
    else:
        raise NotImplementedError(f'Unsupported model: {model_name}')
    return model


def build_resUnet(model_name, config):
    if model_name == 'resnet18':
        from resunet.resunet import Res18_UNet
        n_class = config['n_class']
        return Res18_UNet(n_classes=n_class)
    else:
        raise NotImplementedError(f'Unsupported model: {model_name}')


if __name__ == '__main__':
    model = build_resUnet('resnet18', {'n_class': 2})
    img = torch.rand([1, 3, 256, 256])
    print(model(img).shape)

