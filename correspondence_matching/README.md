## Correspondence Matching

[**Correspondence Matching**](https://github.com/willard-yuan/cvtk/tree/master/correspondence_matching)，局部特征匹配是研究了十几年的课题，目前比较主流且常用的方法除了曾在博客[SIFT Matching with RANSAC](http://yongyuan.name/blog/SIFT(ASIFT)-Matching-with-RANSAC.html)介绍过最近邻/次近邻、RANSAC及其变种方式外，还有霍夫投票、以及基于运动估计的[GMS-Feature-Matcher](https://github.com/JiawangBian/GMS-Feature-Matcher)。

GMS是基于运动统计的方式进行的误匹配点删除，所以要求提取的局部特征数目多，否则GMS失效，另外GMS在实际使用的时候，并不是很鲁棒，小概率出现两个不是同一目标的误匹配。

SVF算法是一种基于霍夫投票的误匹配点剔除方法，实际应用测试，具有很强的鲁棒性。如果在项目中有使用SVF算法，请对其进行引用：

```text
@online{SVF, author = {Yong Yuan}, 
   title = {{SVF} Spatial VeriFication}, 
  year = 2018, 
  url = {https://github.com/willard-yuan/cvtk/tree/master/correspondence_matching}, 
  urldate = {2018} 
 } 
```

如果SVF算法有相应的改进，非常欢迎提交改进的部分。

### 应用场景

Instance Search、Duplicate Image Detection、Image Copy Detection等领域。

### 匹配算法

- `svf.hpp`是匹配算法的主要接口，对角度进行弱几何校验后进行霍夫投票，得到校验后的匹配点后，再删除重复匹配的点对，防止出现一对多或多对一的匹配情况出现。匹配使用的是rootsift。

### 效果测评

应用测试，非常鲁棒。