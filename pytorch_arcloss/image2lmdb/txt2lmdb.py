#!/usr/bin/env python
# encoding: utf-8

import cv2
import six
import os, sys
from PIL import Image
import numpy as np

import lmdb
#import umsgpack
import tqdm
import pyarrow as pa

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

from train_data_flow import TrainDataFlow


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        img, idx = line.split(" ")
        map[img] = idx
    return map


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            print("key", self.keys[index].decode("ascii"))
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)
        imgbuf = unpacked[0]
        label = int(unpacked[1])

        img = np.frombuffer(imgbuf, np.uint8).reshape(299, 299, 3)
        cv2.imwrite('./tmp/' + str(index) + '_' + str(label) + '.jpg', img)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(data_root, file_list, lmdb_dir, write_frequency=5000):
    dataset = TrainDataFlow(data_root, file_list)
    #dataset = ImageFolderWithPaths(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)
    lmdb_path = os.path.join(lmdb_dir, "train.lmdb")
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == '__main__':
    data_root = '/Users/willard/projects/image2lmdb'
    file_list = '/Users/willard/projects/image2lmdb/small_test.txt'
    lmdb_dir = '/Users/willard/projects/image2lmdb/img/lmdb'
    folder2lmdb(data_root, file_list, lmdb_dir)
