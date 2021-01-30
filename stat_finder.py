import os
import pickle
import argparse

from cv2 import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm


# IMG_DIR = "D:/Datasets/COCO - Multilabel/imgs/train"
TARGET_SHAPE = (224, 224)


def get_stats(files, dir=None):
    mean = np.zeros(3)
    var = np.zeros(3)

    print('='*50)
    print('finding sample mean')
    for f in tqdm(files):
        f = f if dir is None else os.path.join(dir, f)
        im = cv2.imread(os.path.join(img_dir, f))
        im = cv2.resize(im, TARGET_SHAPE)/255.
        mean += np.mean(im, axis=(0, 1))
    mean /= len(files)

    print('='*50)
    print('finding sample variance')
    for f in tqdm(files):
        f = f if dir is None else os.path.join(dir, f)
        im = cv2.imread(os.path.join(img_dir, f))
        im = cv2.resize(im, TARGET_SHAPE)/255.
        var += np.mean(im, axis=(0, 1)) - mean
    var /= len(files) - 1

    return mean, var


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('-save_dir')
    parser.add_argument('-train', type=float, help='training portion')
    parser.add_argument('-val', type=float, help='validation portion')
    parser.add_argument('-test', type=float, help='testing portion')

    args = parser.parse_args()
    if not os.path.isdir(args.img_dir):
        raise argparse.ArgumentTypeError(
            f"img_dir {args.img_dir} is not a valid folder")
    if (args.save_dir is not None) and (not os.path.isdir(args.save_dir)):
        raise argparse.ArgumentTypeError(
            f"save_dir {args.save_dir} is not a valid folder")

    train = 0 if args.train is None else args.train
    val = 0 if args.val is None else args.val
    test = 0 if args.test is None else args.test

    assert ((train + val + test) == 1) and all(
        portion > 0 for portion in (train, test, val))

    return args.img_dir, args.save_dir, train, val, test


if __name__ == '__main__':
    img_dir, save_dir, train, val, test = parse_args()

    files = os.listdir(img_dir)
    train_amnt = len(files) * train
    train_amnt = int(train_amnt)
    val_amnt = len(files) * val
    val_amnt = int(val_amnt)
    test_amnt = max(0, int(len(files) - train_amnt - val_amnt))

    all_files = os.listdir(img_dir)
    shuffle(all_files)

    train_files = all_files[:train_amnt]
    val_files = all_files[train_amnt:train_amnt+val_amnt]
    test_files = all_files[-test_amnt:]

    mean, var = get_stats(train_files, img_dir)

    info = {
        'mean': mean,
        'var': var,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files
    }

    with open(os.path.join(save_dir, 'info.pkl'), "wb") as f:
        pickle.dump(info, f)
        f.close()