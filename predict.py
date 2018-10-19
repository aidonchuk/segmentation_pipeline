import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import Compose, Resize, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import SegDataSet
from models.unet import SE_ResNeXt_50
from models_common import AlbuNet
from prepare_train_val import get_test_data
from tta import Nothing, HFlip


def img_to():
    return Compose([
        Resize(128, 128),
        Normalize(p=1)
    ], p=1)


def img_back():
    return Compose([
        Resize(101, 101)
    ], p=1)


def get_model(model_path, model_type='SE_ResNeXt_50'):
    num_classes = 1

    if model_type == 'SE_ResNeXt_50':
        model = SE_ResNeXt_50(num_classes=num_classes)
    if model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()
    return model


def predict(model, from_file_names, batch_size, img_output_path, img_transform_to, img_transform_back, predict_path,
            file_name, draw_predict):
    loader = DataLoader(
        dataset=SegDataSet(from_file_names, transform=img_transform_to, mode='test'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    probs = None
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)
            tta = [Nothing, HFlip]
            r = []
            for cls in tta:
                op = cls()
                ret = op(model, inputs)
                r.append(ret)
            r = np.swapaxes(np.squeeze(do_average(r, np.mean), 1), 0, 2)
            r = np.swapaxes(transform_back(r, img_transform_back), 2, 0)
            if probs is None:
                probs = r
            else:
                probs = np.append(probs, r, 0)

            if draw_predict:
                draw_images(r, paths)

    write_probs(probs, predict_path, file_name)


def write_probs(probs, predict_path, file_name):
    path = predict_path + '/predict/'
    fold_path = Path(path)
    fold_path.mkdir(exist_ok=True, parents=True)

    with open(path + file_name[:file_name.index('.')] + '.pickle', 'wb') as f:
        pickle.dump(probs, f)


def draw_images(batch, paths):
    for i in range(len(batch)):
        as_mask = do_threshold(batch[i], threshold=0.5)
        cv2.imwrite(str(img_output_path) + '/' + paths[i] + '_m.png', np.squeeze(as_mask) * 255)
        print()


def transform_back(image, transform):
    data = {"image": image}
    augmented = transform(**data)
    image = augmented["image"]
    return image


def do_average(data, method):
    z = method(data, 0)
    return z


def do_threshold(v, threshold=0.5):
    v = v.copy()
    v[v >= threshold] = 1
    v[v < 1] = 0

    return v.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='trained_models/SE_ResNeXt_50', help='path to model folder')
    arg('--model_type', type=str, default='SE_ResNeXt_50', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34', 'SE_ResNeXt_50'])
    arg('--output_path', type=str, help='path to save images', default='predict/images')
    arg('--batch-size', type=int, default=4)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--problem_type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=12)

    args = parser.parse_args()

    file_names = get_test_data()
    for fold in [0, 1, 6, 7]:  # 0, 1, 2, 3, 4
        print('Fold: ' + str(fold))
        path_ = str(Path(args.model_path).joinpath('fold_{fold}/'.format(fold=fold)))
        for i in os.listdir(path_):
            temp_path = path_ + '/' + i
            if os.path.isfile(temp_path):
                print('Snapshot: ' + str(i))
                model = get_model(temp_path, model_type=args.model_type)
                print('num file_names = {}'.format(len(file_names)))
                img_output_path = Path(args.output_path)
                img_output_path.mkdir(exist_ok=True, parents=True)

                predict(model, file_names, args.batch_size, img_output_path, img_to(), img_back(), path_, i, False)

'''
                for i in range(r.shape[0]):
                # v = np.fliplr(np.fliplr(np.squeeze(r[i])))
                as_mask = do_threshold(r[i], threshold=0.5)
                cv2.imwrite(str(img_output_path) + '/' + paths[i] + '_m.png', np.squeeze(as_mask) * 255)
            print()

                            for i in range(len(r)):
                    # v = np.fliplr(np.fliplr(np.squeeze(r[i])))
                    as_mask = do_threshold(r[i], threshold=0.5)
                    cv2.imwrite(str(img_output_path) + '/' + paths[i] + str(i) + '_m.png',
                                np.squeeze(as_mask[i]) * 255)


            v = F.sigmoid(outputs).cpu().detach().numpy()

            as_mask = np.squeeze(do_threshold(v[0], threshold=0.5))
            cv2.imwrite(str('predict/images') + '/' + '_m.png',
                        as_mask * 255)


                as_mask = do_threshold(ret, threshold=0.5)
                cv2.imwrite(str(img_output_path) + '_m.png', np.swapaxes(as_mask, 0, 2) * 255)
                # cv2.imwrite(str('predict_train/') + paths[0][:-4] + '_i.png', np.squeeze(img.numpy().astype(np.uint8)))

                def write_probs(probs, fold_name):
                    with open(root_folder + '/fold_' + str(fold_name) + '/probs/' + ''.join(
                            random.choices(string.ascii_letters + string.digits, k=6)).lower() + '.pickle', 'wb') as f:
                        pickle.dump(probs, f)
                def write_submission(df):
                    df = pd.DataFrame.from_records(df, columns=['id', 'rle_mask'])
                    df.to_csv(root_folder + '/predict/submission.csv', index=False)
'''
