import os

import numpy as np
import pandas as pd


def get_split(fold, path='../data/', hem=False):
    df = pd.read_csv(os.path.join(path, 'folds.csv'))

    '''
        f = open(path + 'to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        df = df[df.id != i]
    '''

    # p = read_pseudo(path)

    val = df.loc[df['fold_id'] == fold]
    train = df[df['fold_id'] != fold]

    train_file_names = train.as_matrix(columns=['id'])
    val_file_names = val.as_matrix(columns=['id'])

    # train_file_names = np.concatenate((train_file_names, p))

    if hem:
        hem_df = pd.read_csv(os.path.join(path, 'hem_fold_' + str(fold) + '.csv'))  # .head(5)
        train_file_names = hem_df.as_matrix(columns=['id'])

    return np.squeeze(train_file_names), np.squeeze(val_file_names)


def get_all(fold, path='../data/'):
    df = pd.read_csv(os.path.join(path, 'folds.csv'))
    '''
    f = open(path + 'to_remove_id.txt', 'r')
    x = f.read().split('\n')
    f.close()

    for i in x:
        df = df[df.id != i]
    '''

    val = df.loc[df['fold'] == fold]
    train = df[df['fold'] != fold]

    train_file_names = train.as_matrix(columns=['id'])
    val_file_names = val.as_matrix(columns=['id'])

    return train_file_names, train_file_names


def get_test_data():
    return [x[:-4] for x in os.listdir('../data/test/images/')]  # os.listdir('../data/test/images/')  #


def read_pseudo(path):
    f_names = np.asarray([[i[:-4]] for i in os.listdir(path + '/train/pseudo_labels/images/')])
    return f_names
