import os
import cv2
import numpy as np
import gzip


def load_dataset_from_gz(input_path):
    files = [
        'emnist-balanced-train-labels-idx1-ubyte.gz', 'emnist-balanced-train-images-idx3-ubyte.gz',
        'emnist-balanced-test-labels-idx1-ubyte.gz', 'emnist-balanced-test-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(input_path, fname))  # The location of the dataset.

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def load_dataset_from_images(input_path):
    xs = []
    ys = []
    labels = [f.name for f in os.scandir(input_path) if f.is_dir()]
    for y in labels:
        folder_path = os.path.join(input_path, y)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                # Read the image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    xs.append(img)
                    ys.append(int(y))
    return np.stack(xs), np.stack(ys)


def concat_datasets(x_ds_1, y_ds_1, x_ds_2, y_ds_2):
    x_ds = np.concatenate((x_ds_1, x_ds_2), axis=0)
    y_ds = np.concatenate((y_ds_1, y_ds_2), axis=0)
    return x_ds, y_ds


def filter_dataset_by_labels(x_ds, y_ds, labels):
    indices_to_keep = np.isin(y_ds, labels)

    x_ds = x_ds[indices_to_keep]
    y_ds = y_ds[indices_to_keep]

    return x_ds, y_ds
