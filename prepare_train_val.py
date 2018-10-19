import os

import numpy as np
import pandas as pd


def get_split(fold, path='../data/', hem=False):
    df = pd.read_csv(os.path.join(path, 'folds.csv'))

    val = df.loc[df['fold_id'] == fold]
    train = df[df['fold_id'] != fold]

    train_file_names = train.as_matrix(columns=['id'])
    val_file_names = val.as_matrix(columns=['id'])

    if hem:
        hem_df = pd.read_csv(os.path.join(path, 'hem_fold_' + str(fold) + '.csv'))  # .head(5)
        train_file_names = hem_df.as_matrix(columns=['id'])

    return np.squeeze(train_file_names), np.squeeze(val_file_names)


def get_test_data():
    return [x[:-4] for x in os.listdir('../data/test/images/')]  # os.listdir('../data/test/images/')  #
