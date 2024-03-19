from torch import nn
from torchvision.models import resnet18, vgg16

from .model_classifier import UNet_Classifier
from .model_segmentation import UNet


def build_cls_model(model_name, config):
    n_channel = config['n_channel']
    n_class = config['n_class']
    if model_name == 'unet_classifier':
        model = UNet_Classifier(n_channel, n_class)
    elif model_name == 'resnet18':
        # resnet18 channel must be 3
        model = resnet18(pretrained=True)
        features = model.fc.in_features
        model.fc = nn.Linear(features, n_class)

    elif model_name == 'vgg16':
        model = vgg16(pretrained=True)
        features = model.fc.in_features
        model.fc = nn.Linear(features, n_class)

    else:
        raise NotImplementedError(f'Unsupported model: {model_name}')
    return model


def build_seg_model(model_name, config):
    n_channel = config['n_channel']
    n_class = config['n_class']
    if model_name == 'unet':
        model = UNet(n_channels=n_channel, n_classes=n_class)
    else:
        raise NotImplementedError(f'Unsupported model: {model_name}')
    return model