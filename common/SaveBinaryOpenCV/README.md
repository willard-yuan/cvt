SaveBinaryOpenCV
===========

This is expantion of BinaryCvMat that saves and Loads cv::Mat as a binary file.

https://github.com/takmin/BinaryCvMat


cv::FileStorage is not so fast to save cv::Mat of large size, because it writes datas as ASCII.
So the above code saves cv::Mat data as binary.


This project saves not only cv::Mat but cv::Size, cv::Point, cv::KeyPoint, and std::vector<cv::KeyPoint> can be saved as a binary file .

It requires OpenCV 2.0 or later.

You can use these functions by adding "SaveBinaryCV.h" and "SaveBinaryCV.cpp" to your code.

Please read "SaveBinaryCV.h".  You can find how to use these functions.

For example, if you want to save cv::Mat and std::vector<cv::KeyPoint> in one file, you can write save and load function as follows:

```
void Save(const std::string& filename, const cv::Mat& A, const std::vector<cv::KeyPoint>& B)
{
  std::ofstream ofs(filename, std::ios::binary);
  writeMatBinary(ofs, A);
  writeKeyPointsBinary(ofs, B);
}


void Load(const std::string& filename, cv::Mat& A, std::vector<cv::KeyPoint>& B)
{
  std::ifstream ifs(filename, std::ios::binary);
  readMatBinary(ifs, A);
  readKeyPointBinary(ifs, B)
}
```
