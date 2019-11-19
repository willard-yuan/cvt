
#ifndef opencvUtils_hpp
#define opencvUtils_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>


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
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_reduced_dim);
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

#endif /* opencv_utils_hpp */
