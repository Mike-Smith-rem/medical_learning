import random
import numpy as np
import torch

from dataset.busi_dataset import classifer_busi_dataloader, segment_busi_dataLoader
from dataset.cifer10_dataset import classifier_cifer10_dataset
from model.build_model import build_cls_model, build_seg_model
from train.cls_train import cls_train
from train.seg_train import seg_train
from train.utils.logger import Logging

args = {
    'type': 'classifier',
    'dataset': 'busi',
    'model': 'unet_classifier',
    'load_normal_data': True,
    'epochs': 200,
    'bs': 16,
    'seed': 2024,
    'criterion': 'CrossEntropyLoss',
    'optimizer': 'adam',
    'lr_scheduler': 'StepLR'
}
other = {
    'alpha': [1, 1],
    'gamma': 2,
    'num_classes': 2,

    'mode': None,
    'mode_path': None,

    'lr': 0.0001,
    'weight_decay': 5e-4,
    'momentum': 0.9,

    'step_size': 5,
    'lr_gamma': 0.9
}

debug = {
    'dataset': True,
    'loss': False,
}


def init():
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])

    if args['type'] == 'segment':
        model = build_seg_model(args['model'], config={'n_class': 2, 'n_channel': 3})
        logger = Logging(f"./logs/{args['model']}_{args['criterion']}_{args['load_normal_data']}_{args['dataset']}")
        if args['mode'] is not None:
            model = build_seg_model(args['model'], config={'n_class': 2, 'n_channel': 3},
                                    mode=args['mode'], mode_path=args['mode_path'])
            logger = Logging(f"./logs/{args['model']}_{args['criterion']}_"
                             f"{args['load_normal_data']}_{args['dataset']}_"
                             f"{args['mode']}_{args['mode_path']}")
        train_loader, test_loader = segment_busi_dataLoader(args['load_normal_data'], args['bs'], args['model'])
        seg_train(args, other, model, train_loader, test_loader, logger)
        logger.save_file("finish training.", None)
        logger.save_config(args)
        logger.save_config(other)
        # train_again

    elif args['type'] == 'classifier':
        model = build_cls_model(args['model'], config={'n_class': 3, 'n_channel': 3})
        logger = Logging(f"./logs/{args['model']}_{args['criterion']}_{args['load_normal_data']}_{args['dataset']}")
        if args['mode'] is not None:
            model = build_cls_model(args['model'], config={'n_class': 3, 'n_channel': 3},
                                    mode=args['mode'], mode_path=args['mode_path'])
            logger = Logging(f"./logs/{args['model']}_{args['criterion']}_"
                             f"{args['load_normal_data']}_{args['dataset']}_"
                             f"{args['mode']}_{args['mode_path']}")
        train_loader, test_loader = classifer_busi_dataloader(args['load_normal_data'], args['bs'],
                                                              args['model'], debug=debug['dataset'])
        if args['dataset'] == 'cifar10':
            model = build_cls_model(args['model'], config={'n_class': 10, 'n_channel': 3})
            train_loader, test_loader = classifier_cifer10_dataset()
        cls_train(args, other, model, train_loader, test_loader, logger)
        logger.save_file("finish training.", None)
        logger.save_config(args)
        logger.save_config(other)
    else:
        pass


if __name__ == '__main__':
    debug['dataset'] = True
    args['criterion'] = 'CrossEntropyLoss'

    # 1. 实现Unet分割过程, load_normal_data is true
    args['type'] = 'segment'
    args['dataset'] = 'busi'
    args['model'] = 'unet'
    args['load_normal_data'] = True
    init()

    # 2. 实现Unet_classifier分类过程， load_normal_data is false
    args['type'] = 'classifier'
    args['dataset'] = 'busi'
    args['model'] = 'unet_classifier'
    args['load_normal_data'] = True
    init()

    # 3. 实现resnet分类过程，load_normal_data is false
    # args['type'] = 'classifier'
    # args['dataset'] = 'busi'
    # args['model'] = 'resnet18'
    # args['load_normal_data'] = False

    # 4. 实现Unet_classifier, 对cifar10的分类
    # args['type'] = 'classifier'
    # args['dataset'] = 'cifar10'
    # args['model'] = 'unet_classifier'
    # init()

    # 5. 实现resnet18, 对cifar10的分类
    # args['type'] = 'classifier'
    # args['dataset'] = 'cifar10'
    # args['model'] = 'resnet18'
    # init()
    # # ??
    # # 6. 实现Unet encoder冻结，三分类帮助三分割
    # args['type'] = 'segment'
    # args['mode'] = 'cls_to_seg'
    # args['dataset'] = 'busi'
    # args['model'] = 'unet'
    # args['mode_path'] = './logs/unet_classifier_CrossEntropyLoss_' \
    #                     'True_busi/UNet_Classifier_test_acc_0.8080808080808081.pth'
    # init()
    # 7. 实现Unet encoder冻结，二分割帮助二分类
    # args['type'] = 'classifier'
    # args['mode'] = 'seg_to_cls'
    # args['dataset'] = 'busi'
    # args['model'] = 'unet_classifier'
    # args['mode_path'] = './logs/unet_CrossEntropyLoss_True_busi/UNet_test_dice_0.8335910588502884.pth'
    # init()
    # 8. 实现联邦场景下的cifar10分类(iid)

    # 9. 实现联邦场景下的busi分类(iid)
