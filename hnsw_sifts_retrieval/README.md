## HNSW SIFTs Retrieval

[**hnsw_sifts_retrieval**](https://github.com/willard-yuan/mykit/tree/master/cvkit/hnsw_sifts_retrieval)，是一个已直接索引SIFT并通过SIFT匹配点数排序的检索应用，在使用SIFT点数排序的时候，对SIFT匹配的点数做了弱几何校验，剔除误匹配的点数。

### 数据适应规模

适合在小中型数据规模上检索。

### 文件说明

- [hnswlib](https://github.com/nmslib/hnsw)，对于SIFT特征点的索引，采用HNSW进行索引，HNSW是一种以图方式构建的ANN搜索方法，选择该方法的理由是该方法在ANN搜索方法里，取得的召回率是很优秀的。
- `makeSIFTs.cpp`，对库中的图片提取rootsift特征，并将描述子以及几何信息保存在一个文件中。
- `makeIdx.cpp`，对提取的rootsift特征构建索引。
- `makeSearch.cpp`，对输入的查询图片进行检索。考虑到需要对每个SIFT描述子都需要做1次查询，为保证效率，提取SIFT的时候，设置点数不超过128个。具体可以根据实际使用调整该值。

### 效果测评

测试了少量的case，效果还不错。