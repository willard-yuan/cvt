# -*- coding: UTF-8 -*-
# python3.7

import struct
import random
import numpy as np
from sklearn.preprocessing import normalize as sknormalize


dim_feat = 64
qfeats_path = 'embed_v2.txt'

qfeats_file = open(qfeats_path, 'r')
qcontent = qfeats_file.readlines()

def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

num = len(qcontent)

ids = []
featsArray = np.zeros((num, dim_feat),dtype = np.float32)
count = 0


# 写入查询文件
for i, line in enumerate(qcontent):
    feat = [float(value) for value in line.strip().split(',')[1].split('\x02')]
    id_ = line.strip().split(',')[0]
    if(len(feat) != dim_feat):
        print("error: %s" % id_)
        continue
    if count < num:
        featsArray[count,:] = np.array(feat, dtype = np.float32)
        ids.append(id_)
        count += 1
    else:
        break
qfeats_file.close()

del qcontent

featsArray = normalize(featsArray)

# 库特征写入二进制
db_bin_file = open('embed_v2.bin', 'wb')
line_binary = struct.pack('i', count)
db_bin_file.write(line_binary)
for i in range(count):
    tmpFeat = list(featsArray[i,:])
    tmpId = ids[i]
    #format_ = 'i' + 's'*len(tmpId)
    #line_binary = struct.pack(format_, len(tmpId), *tmpId)
    line_binary = struct.pack('i', len(tmpId))
    db_bin_file.write(line_binary)
    for str_ in tmpId:
        line_binary = struct.pack('s', bytes(tmpId.encode('utf-8')))
        db_bin_file.write(line_binary)
    #line_binary = struct.pack(format_, len(tmpId), *tmpId),
    #db_bin_file.write(line_binary)
    #format_ = 'i' + 'f'*dim_feat
    #line_binary = struct.pack(format_, dim_feat, *tmpFeat)
    #db_bin_file.write(line_binary)
    line_binary = struct.pack('i', dim_feat)
    db_bin_file.write(line_binary)
    line_binary = struct.pack('f'*dim_feat, *tmpFeat)
    db_bin_file.write(line_binary)
db_bin_file.close()
print("database finished writing to binary file")
