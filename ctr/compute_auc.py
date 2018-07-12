
import os, sys
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

def scoreAUC(labels, probs):
    i_sorted = sorted(range(len(probs)),key=lambda i: probs[i],
                      reverse=True)
    auc_temp = 0.0
    TP = 0.0
    TP_pre = 0.0
    FP = 0.0
    FP_pre = 0.0
    P = 0
    N = 0
    last_prob = probs[i_sorted[0]] + 1.0
    for i in range(len(probs)):
        if last_prob != probs[i_sorted[i]]: 
            auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0        
            TP_pre = TP
            FP_pre = FP
            last_prob = probs[i_sorted[i]]
        if labels[i_sorted[i]] == 1:
            TP = TP + 1
        else:
            FP = FP + 1
    auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0
    auc = auc_temp / (TP * FP)
    return auc

retfile = open('ret_ctr.txt', 'r')
content = retfile.readlines()

labels = []
probs = []
for line in content[:-1]:
    line = line.strip().split(' ')
    label = int(line[1])
    label_pre = int(line[2])
    prob = float(line[3])
    if label_pre == 0:
        prob = 1.0 - prob
    labels.append(label)
    probs.append(prob)

auc = scoreAUC(labels, probs)
print auc

print roc_auc_score(np.asarray(labels), np.asarray(probs))
