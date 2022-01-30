## 全局特征

通过OpenCV提取Pytorch、Tensorflow、Caffe等模型的特征。下面是OpenCV提取Caffe模型的应用，模块详见[opencv_dnn_feature](https://github.com/willard-yuan/cvtk/tree/master/opencv_dnn_feature)。在编译之前，修改[CMakeLists.txt](https://github.com/willard-yuan/cvtk/blob/master/opencv_dnn_feature/CMakeLists.txt)的`OpenCV_DIR`，然后通过下面方式即可编译：

```sh
cd opencv_dnn_feature
mkdir build
cmake ..
make
```
