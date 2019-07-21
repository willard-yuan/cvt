
#ifndef opencvUtils_hpp
#define opencvUtils_hpp

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>

/**********************************************
 * OpenCV functions
 **********************************************/


/*
 Get type of cv::Mat
 */
std::string type2str(int type)
{
    std::string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}


// Convert vector of vector to OpenCV Mat
cv::Mat convert2Mat(std::vector<std::vector<float> > &vecs){
    cv::Mat mat_matrix((int)vecs.size(), (int)vecs.at(0).size(), CV_32FC1);
    for(int i=0; i<mat_matrix.rows; ++i)
        for(int j=0; j<mat_matrix.cols; ++j)
            mat_matrix.at<float>(i, j) = vecs.at(i).at(j);
    return mat_matrix;
}

/* Train PCA using OpenCV
 Input:
     data:             n*d
     num_reduced_dim:  reduded number
     filename:         path of saved
 */
void trainPCA(cv::Mat &data, int num_reduced_dim, std::string &filename){
    if(data.cols <= 0 || data.rows <= 0){
        std::cout << "data is empty, training is failed" << std::endl;
    }
    //PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1.0);
    std::cout << "start traing PCA model" << std::endl;
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, num_reduced_dim);
    std::cout << "traing PCA model is finished" << std::endl;
    std::cout << "start writing PCA model" << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    pca.write(fs);
    std::cout << "writing PCA model is finished" << std::endl;
}


/* Load PCA model
Input:
    filename:         path of saved
 */
cv::PCA loadPCA(std::string &filename){
    cv::PCA pca;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    pca.read(fs.root());
    return pca;
}


/* L2 normalize of opencv Mat,
 * each row is a sample
 */
cv::Mat l2norm(cv::Mat data){
    for(int i = 0; i < data.rows; i++){
        cv::Mat norm_mat;
        cv::sqrt(data.row(i)*data.row(i).t(), norm_mat);
        float denorm_v = std::max((float)1e-12, (float)norm_mat.at<float>(0, 0));
        for (int j = 0; j < data.cols; j++){
            data.at<float>(i, j) = data.at<float>(i, j)/denorm_v;
        }
    }
    return data;
}

#endif /* opencvUtils_hpp */
