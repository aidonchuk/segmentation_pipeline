import argparse
from pathlib import Path

import torch
import torch.backends.cudnn
import torch.backends.cudnn as cudnn
from albumentations import (
    HorizontalFlip,
    Normalize,
    Compose,
    Resize, RandomBrightness, ShiftScaleRotate, ElasticTransform, GridDistortion, OneOf)
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils
from dataset import SegDataSet
from loss import FocalLovasz, RFocalLovaszJaccard, RobustFocalLoss2d, FocalJaccard, LossBinary
from lovasz_losses import LovaszHingeLoss, LovaszBCE
from models.LinkNet import LinkNeXt
from models.duc_hdc import ResNet50_DUCHDC
from models.gcn import GCN
from models.unet import SE_ResNeXt_50, DenseNet161, Incv3
from models_common import UNet11, UNet16, UNet, AlbuNet, LinkNet34, SeRes50NextHyper
from prepare_train_val import get_split
from validation import validation_binary

torch.set_default_tensor_type('torch.cuda.FloatTensor')

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'SeRes50NextHyper': SeRes50NextHyper,
               'SE_ResNeXt_50': SE_ResNeXt_50,
               'DenseNet161': DenseNet161,
               'LinkNeXt': LinkNeXt,
               'LinkNet34': LinkNet34,
               'GCN': GCN,
               'Incv3': Incv3,
               'ResNet50_DUCHDC': ResNet50_DUCHDC}

losses = {
    'lava': LovaszHingeLoss(),
    'bce': BCEWithLogitsLoss(),
    'bce_jaccard': LossBinary(),
    'bce_lava': LovaszBCE(bce_weight=0.1),
    'focal': RobustFocalLoss2d(),
    'focal_lava': FocalLovasz(focal_weight=0.3),
    'focal_jaccard': FocalJaccard(),
    'focal_lava_jaccard': RFocalLovaszJaccard(jaccard_weight=0.15, focal_weight=0.15)
}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device_ids', type=str, default='0,1,2,3', help='For example 0,1 to run on two GPUs')
    arg('--requires_grad', type=bool, default=False, help='freez encoder')
    arg('--start_epoch', type=str, default='0', help='start epoch emp 21')
    arg('--rop_step', type=int, default=6, help='reduce on plateu step')
    arg('--hem_sample_count', type=int, default=0, help='hard example sample count')
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0,1', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=256)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0003)
    arg('--workers', type=int, default=20)
    arg('--loss', type=str, default='bce_lava')
    arg('--optim', type=str, default='adam')
    arg('--scheduler', type=str, default='rop')
    arg('--early_stop_patience', type=int, default=1000)
    arg('--save_best_count', type=int, default=6)

    arg('--model', type=str, default='SE_ResNeXt_50', choices=moddel_list.keys())

    args = parser.parse_args()
    print(args)
    num_classes = 1

    fold_path = Path(args.root + "/" + args.model + '/fold_' + str(args.fold))
    fold_path.mkdir(exist_ok=True, parents=True)

    model_name = moddel_list[args.model]
    model = model_name(num_classes=num_classes, pretrained=True, requires_grad=args.requires_grad)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    loss = losses[args.loss]
    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, batch_size=1):
        return DataLoader(
            dataset=SegDataSet(file_names, transform=transform),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            Resize(64, 64),
            OneOf([
                GridDistortion(),
                ElasticTransform(),
            ], p=0.0),
            RandomBrightness(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=5, p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            Resize(64, 64),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1),
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), batch_size=args.batch_size)
    valid = validation_binary

    optimizers = {
        'adam': optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000001),
        'rmsprop': optim.RMSprop(model.parameters(), lr=args.lr),
        'sgd': optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.9)
    }

    optimizer = optimizers[args.optim]

    scheduler = {
        'co': CosineAnnealingLR(optimizer, T_max=6, eta_min=0.001),
        'rop': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.rop_step, verbose=True)
    }

    utils.train(
        optimizer=optimizer,
        scheduler=scheduler[args.scheduler],
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
