import os
import lmdb
from txt2lmdb import ImageFolderLMDB
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

def main(path):
    #transform = Resize((224, 224))
    #dst = ImageFolderLMDB(path, transform, None)
    dst = ImageFolderLMDB(path)
    trainloader = DataLoader(dst, shuffle=True, batch_size=6, drop_last=True)
    for i, data in enumerate(trainloader):
        print(data[0].shape, data[1])


if __name__ == '__main__':
    path = "/Users/willard/projects/image2lmdb/img/lmdb/train.lmdb"
    main(path)
