
#ifndef general_h
#define general_h

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

template <typename T>
cv::Mat_<T> vec2cvMat_2D(std::vector< std::vector<T> > &inVec){
    int rows = static_cast<int>(inVec.size());
    int cols = static_cast<int>(inVec[0].size());
    
    cv::Mat_<T> resmat(rows, cols);
    for (int i = 0; i < rows; i++){
        resmat.row(i) = cv::Mat(inVec[i]).t();
    }
    return resmat;
}

template <typename T>
arma::mat vec2mat(std::vector<std::vector<T>> &vec){
    int col = (int)vec.size();
    int row = (int)vec[0].size();
    arma::mat A(row, col, arma::fill::zeros);
    for(int i = 0; i < col; i++){
        for(int j=0; j < row; j++){
            A(j, i) = vec[i][j];
        }
    }
    return A;
}

// 将特征拼成一位数组
template <typename T>
T * vectors2OneArray(std::vector<std::vector<T>> &descs){
    T * descsToOneArray = (T *)malloc(sizeof(T)*descs.size()*128);
    for(int i = 0; i < descs.size(); i++){
        for(int j = 0; j < 128; j++){
            descsToOneArray[i*128+j] = descs[i].at(j);
            //std::cout << std::setiosflags(std::ios::fixed);
            //std::cout << std::setprecision(6) << descsInOne[i*128 +j] << ", ";
        }
        //std::cout << "\n" << std::endl;
    }
    return descsToOneArray;
}

#endif /* general_h */
