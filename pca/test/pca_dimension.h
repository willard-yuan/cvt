
#ifndef pca_dimension_h
#define pca_dimension_h

#include <stdio.h>
#include <opencv2/opencv.hpp>

const int dimDL = 1024;

class PCAModel{
public:
    PCAModel(std::string &filename){
        loadPCA(filename);
    };
    cv::Mat reductDimension(const float* data, int queryNum, int dim);
    ~PCAModel(){};
    
private:
    void loadPCA(std::string &filename);
    cv::Mat convertToMat(const float* data, int queryNum, int dim);
    cv::PCA pca_;
};

cv::Mat PCAModel::convertToMat(const float* data, int queryNum, int dim) {
    cv::Mat matAngles(queryNum, dim, CV_32FC1);
    for (int i = 0; i < matAngles.rows; ++i) {
        for (int j = 0; j < matAngles.cols; ++j) {
            matAngles.at<float>(i, j) = *(data + i * dim + j);
        }
    }
    return matAngles;
}

void PCAModel::loadPCA(std::string &filename){
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    // opencv 3.2
    cv::read(fs.root()["vectors"], pca_.eigenvectors);
    cv::read(fs.root()["values"], pca_.eigenvalues);
    cv::read(fs.root()["mean"], pca_.mean);
    
    // opencv 3.3
    //pca_.read(fs.root());
}

cv::Mat PCAModel::reductDimension(const float* data, int queryNum, int dim){
    cv::Mat reduceMat = pca_.project(convertToMat(data, queryNum, dim));
    // L2 normalize
    for(int i = 0; i < reduceMat.rows; i++){
        cv::Mat norn_mat = reduceMat.row(i)*reduceMat.row(i).t();
        float denom_v = std::max(1e-12, sqrt(norn_mat.at<float>(0, 0)));
        for (int j = 0; j < dimDL; j++){
            reduceMat.at<float>(i, j) = reduceMat.at<float>(i, j)/sqrt(denom_v);
        }
    }
    return reduceMat;
}

#endif /* pca_dimension_h */
