import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ISZ = 101


def star_fold_by_mask_size():
    train = pd.read_csv(os.path.join('../data/', 'train.csv'))

    l = []
    for index, row in train.iterrows():
        id = row['id']
        m = row['rle_mask']
        if isinstance(m, float):
            l.append(0)
            continue
        m = rleToMask(m, 101, 101)
        l.append(np.count_nonzero(m))
    train['mask_len'] = l
    train = train.drop('rle_mask', axis=1)
    train.to_csv(os.path.join('../data/', 'train_mask_len.csv'), index=False)


def rleToMask(rleString, height, width):
    rows, cols = height, width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


def prepare_fold_stra_mask_len():
    n_fold = 5
    train = pd.read_csv(os.path.join('../data/', 'train_mask_len.csv'))

    f = open('../data/to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        train = train[train.id != i]

    train.sort_values('mask_len', inplace=True)
    train.drop('mask_len', axis=1, inplace=True)
    train['fold'] = (list(range(n_fold)) * train.shape[0])[:train.shape[0]]
    print(train.head())
    train.to_csv(os.path.join('../data/', 'folds.csv'), index=False)


def prepare_fold_stra_rnd_0():
    n_fold = 15
    train = pd.read_csv(os.path.join('../data/', 'train.csv'))
    # train.dropna(inplace=True)
    f = open('../data/to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        train = train[train.id != i]

    train = train.sample(frac=1).reset_index(drop=True)
    train['fold'] = (list(range(n_fold)) * train.shape[0])[:train.shape[0]]
    train = train.drop('rle_mask', axis=1)
    print(train.head())
    train.to_csv(os.path.join('../data/', 'folds.csv'), index=False)


def prepare_fold_stra_rnd_1():
    n_fold = 5
    train = pd.read_csv(os.path.join('../data/', 'train.csv'))

    f = open('../data/to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        train = train[train.id != i]

    train = train.sample(frac=1).reset_index(drop=True)
    train['fold'] = (list(range(n_fold)) * train.shape[0])[:train.shape[0]]
    print(train.head())
    train.to_csv(os.path.join('../data/', 'folds.csv'), index=False)


def prepare_fold_stra_depth():
    n_fold = 5
    depths = pd.read_csv(os.path.join('../data/', 'depths.csv'))
    train_rle = pd.read_csv(os.path.join('../data/', 'train.csv'))
    train_rle.drop('rle_mask', axis=1, inplace=True)
    depths = depths.merge(train_rle, on=['id', 'id'])
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)

    f = open('../data/to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        depths = depths[depths.id != i]

    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    print(depths.head())
    depths.to_csv(os.path.join('../data/', 'folds.csv'), index=False)


def fold_400():
    n_fold = 5
    depths = pd.read_csv(os.path.join('../data/', 'depths.csv'))
    depths = depths[depths['z'] < 400]

    train_rle = pd.read_csv(os.path.join('../data/', 'train.csv'))
    train_rle.drop('rle_mask', axis=1, inplace=True)
    depths = depths.merge(train_rle, on=['id', 'id'])

    f = open('../data/to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        depths = depths[depths.id != i]

    train = depths.sample(frac=1).reset_index(drop=True)
    train['fold'] = (list(range(n_fold)) * train.shape[0])[:train.shape[0]]
    print(train.head())
    train.to_csv(os.path.join('../data/', 'folds_less_400.csv'), index=False)


def create_mosaic():
    df = pd.read_csv(os.path.join('../data/', 'folds.csv'))
    for i in range(0, 5):
        v = df.loc[df['fold'] == i].as_matrix(columns=['id'])
        print()


def calc_means_std():
    df = pd.read_csv(os.path.join('../data/', 'folds.csv'))

    m1 = []
    m2 = []
    m3 = []
    images = []
    for v in tqdm(df['id'].values):
        image = cv2.imread(str('../data/' + 'train' + '/images/' + v + '.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        m1.append(image[:, :, 0])
        m2.append(image[:, :, 1])
        m3.append(image[:, :, 2])

    print(np.mean(m1))
    print(np.mean(m2))
    print(np.mean(m3))
    print(np.std(m1))
    print(np.std(m2))
    print(np.std(m3))


def prepare_fold_stra_depth_len():
    n_fold = 10
    depths = pd.read_csv(os.path.join('../data/', 'depths.csv'))
    train_rle = pd.read_csv(os.path.join('../data/', 'train.csv'))
    train_rle.drop('rle_mask', axis=1, inplace=True)
    depths = depths.merge(train_rle, on=['id', 'id'])
    train = pd.read_csv(os.path.join('../data/', 'train_mask_len.csv'))
    depths = depths.merge(train, on=['id', 'id'])

    depths.sort_values(['z', 'mask_len'], inplace=True)

    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    print(depths.head())
    depths.to_csv(os.path.join('../data/', 'folds.csv'), index=False)


prepare_fold_stra_depth_len()
# prepare_fold_stra_depth()
# prepare_fold()
# create_mosaic()

# prepare_fold_stra_rnd_0()
# fold_400()
