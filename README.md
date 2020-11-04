## CVTK简介

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](../LICENSE)

**CVTK, a Computer Vision ToolKit**. 

CVTK以个人计算机视觉实践经验为基础，旨在构建高效的计算机视觉常用工具集。

## CVTK应用

- [**HNSW SIFTs Retrieval**](https://github.com/willard-yuan/cvtk/tree/master/hnsw_sifts_retrieval)，一个已直接索引SIFT并通过SIFT匹配点数排序的检索应用，在使用SIFT点数排序的时候，对SIFT匹配的点数做了弱几何校验，剔除误匹配的点数。详细在[doc](https://github.com/willard-yuan/cvtk/tree/master/hnsw_sifts_retrieval)里有说明。
- [**Correspondence Matching**](https://github.com/willard-yuan/cvtk/tree/master/correspondence_matching). 局部特征匹配是研究了十几年的课题，目前比较主流且常用的方法除了曾在博客[SIFT Matching with RANSAC](http://yongyuan.name/blog/SIFT(ASIFT)-Matching-with-RANSAC.html)介绍过最近邻/次近邻、RANSAC及其变种方式外，还有霍夫投票、以及基于运动估计的[GMS-Feature-Matcher](https://github.com/JiawangBian/GMS-Feature-Matcher)。SVF算法是一种基于霍夫投票的误匹配点剔除方法，实际应用测试，具有很强的鲁棒性。
- [**Covdet**](https://github.com/willard-yuan/cvtk/tree/master/covdet), C++ API for covdet of VLFeat，read [doc](https://github.com/willard-yuan/cvtk/tree/master/covdet) in details。
- [**PCA Train_and Project**](https://github.com/willard-yuan/cvtk/tree/master/pca_train_project)，详细在[doc](https://github.com/willard-yuan/cvtk/tree/master/pca_train_project)有说明。
- [**Brute Force Search**](https://github.com/willard-yuan/cvtk/tree/master/brute_force_search)，大规模最近邻暴力搜索C++实现。
- [**TensorFlow Extract Feature with CPP API**](https://github.com/willard-yuan/cvtk/tree/master/tf_extract_feat)，TensorFlow载入PB模型，提取特征代码，C++实现。
- [**LibTorch Extract Feature with CPP API**](https://github.com/willard-yuan/cvtk/tree/master/libtorch_extract_feat)，Libtorch载入pt模型，提取特征代码（改成分类等类似），C++实现。
- [**scale_quantization**](https://github.com/willard-yuan/cvtk/tree/master/scale_quantization), 标量量化，用于排序上需要获取topK的embedding的场景。
