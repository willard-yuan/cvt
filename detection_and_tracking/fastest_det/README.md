# FastestDet

使用OpenCV部署FastestDet，包含C++和Python两种版本的程序：

> 在知乎上看到一篇在文章《FastestDet: 比yolo-fastest更快！更强！更简单！全新设计的超实时Anchor-free目标检测算法》，
于是我就导出onnx文件，编写了使用OpenCV部署FastestDet，依然是包含C++和Python两种版本的程序。
.onnx文件很小，只有960kb，不超过1M的。适合应用到对实时性要求高的场景里。

## 编译

```bash
cd  fastest_det
mkdir build && cd build
cmake ..
make -j4
./fastest_det_demo
```
	
