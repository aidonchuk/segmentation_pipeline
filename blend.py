import os
import pickle

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_train_val import get_test_data
from rle import rle_encoding


def main():
    v = []
    # for fold in [0, 1, 2, 3 , 4]:  # , 1, 2, 3 , 4
    #    with open('subs/' + str(fold) + '_probs_fold.pickle', 'rb') as f:
    #        data_new = pickle.load(f)
    #        v.append(data_new)

    for i in tqdm(os.listdir('subs/')):
        with open('subs/' + i, 'rb') as f:
            data_new = pickle.load(f)
            v.append(data_new)

    to_df = []
    v = np.asanyarray(v)
    all = np.mean(v, axis=0)
    write_probs(all)
    files = get_test_data()
    for i in tqdm(range(len(files))):
        as_mask = post_process(all[i], threshold=0.5).astype(np.uint8)
        file = files[i][:-4]

        mask_sum = np.sum(as_mask)
        mask_ = as_mask
        # if 30 >= mask_sum > 1:
        #    mask_[mask_ > 0] = 0
        to_df.append((file, rle_encoding(mask_)))
        if True:
            cv2.imwrite(str('predict/images/') + file + '_m.png', as_mask * 255)

    write_submission(to_df)


def write_probs(probs):
    with open('subs/probs_fold_blend.pickle', 'wb') as f:
        pickle.dump(probs, f)


def write_submission(df):
    df = pd.DataFrame.from_records(df, columns=['id', 'rle_mask'])
    df.to_csv('subs/blend.csv', index=False)


def post_process(v, threshold=0.3):
    v = v.copy()
    v[v > threshold] = 1
    v[v < 1] = 0

    blur = ((3, 3), 1)
    erode_ = (5, 5)
    dilate_ = (3, 3)

    # r = cv2.dilate(
    #    cv2.erode(cv2.GaussianBlur(v, blur[0], blur[1]), np.ones(erode_)),
    #    np.ones(dilate_))

    # r[r > 0] = 1
    # v[v > 1] = 0
    return v


main()
