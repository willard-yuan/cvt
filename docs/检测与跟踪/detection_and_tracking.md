## 标量量化

Scalar Quantization标量量化，分为三个过程：
- training过程：主要是训练encode过程，需要的一些参数，这些的参数，主要是每1维对应的最大值、最小值；
- encode过程：将float向量量化为int8向量（int8是其中一种数据压缩形式，还有4比特之类的，这里主要以8比特说明原理）；
- decode过程：将int8向量解码为float向量；

![](http://yongyuan.name/imgs/posts/scalar-quantization-encode-decode1.jpg)
