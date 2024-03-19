import random
import numpy as np
import torch

from dataset.busi_dataset import classifer_busi_dataloader, segment_busi_dataLoader
from model.build_model import build_cls_model, build_seg_model
from train.cls_train import cls_train
from train.seg_train import seg_train
from train.utils.logger import Logging

args = {
    'type': 'segment',
    'dataset': 'busi',
    'model': 'unet',
    'load_normal_data': False,
    'epochs': 200,
    'bs': 16,
    'seed': 2024,
    'criterion': 'FocalLoss',
    'optimizer': 'adam',
    'lr_scheduler': 'StepLR'
}
other = {
    'alpha': [1, 1],
    'gamma': 2,
    'num_classes': 2,

    'lr': 0.0001,
    'weight_decay': 5e-4,
    'momentum': 0.9,

    'step_size': 5,
    'lr_gamma': 0.9
}

if __name__ == '__main__':
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])

    if args['type'] == 'segment':
        model = build_seg_model(args['model'], config={'n_class': 2, 'n_channel': 3})
        logger = Logging(f"./logs/{args['model']}")
        train_loader, test_loader = segment_busi_dataLoader(args['load_normal_data'], args['bs'], args['model'])
        seg_train(args, other, model, train_loader, test_loader, logger)
        logger.save_file("finish training.", None)
        logger.save_config(args)
        logger.save_config(other)

        # train_again

    elif args['type'] == 'classifier':
        model = build_cls_model(args['model'], config={'n_class': 3, 'n_channel': 3})
        logger = Logging(f"./logs/{args['model']}")
        train_loader, test_loader = classifer_busi_dataloader(args['load_normal_data'], args['bs'], args['model'])
        cls_train(args, other, model, train_loader, test_loader, logger)
        logger.save_file("finish training.", None)
        logger.save_config(args)
        logger.save_config(other)
    else:
        pass
