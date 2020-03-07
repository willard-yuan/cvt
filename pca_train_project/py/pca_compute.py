import cv2
import numpy as np
from sklearn.preprocessing import normalize as sknormalize
from sklearn.preprocessing import normalize as sknormalize

def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

fs = cv2.FileStorage("pca_256_500w.yml", cv2.FILE_STORAGE_READ)

eigenvectors = fs.getNode("vectors").mat()
values = fs.getNode("values").mat()
mean = fs.getNode("mean").mat()

dim_feat = 2048
qfeats_path = 'part-00000-2048_1'

qfeats_file = open(qfeats_path, 'r')
qcontent = qfeats_file.readlines()

num = len(qcontent)

feats = []
names = []
for line in qcontent:
    tmplist = line.split(',')
    name = tmplist[0]
    feat = [float(value) for value in tmplist[1:]]
    feats.append(feat)
    names.append(name)

featArray = np.array(feats)

print(mean.shape)
print(eigenvectors.shape)
#print(eigenvectors)

data_compressed = cv2.PCAProject(featArray, mean, eigenvectors)
data_compressed_normal = normalize(data_compressed, copy=False)

for i, name in enumerate(names):
    print(name + ' ' + ' '.join([str(value) for value in data_compressed_normal[i]]))
