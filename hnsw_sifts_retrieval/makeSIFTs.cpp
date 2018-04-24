
#include <map>
#include <string>
#include <glob.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

typedef unsigned char uint8;


void writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
{
    int type = out_mat.type();
    ofs.write((const char*)(&out_mat.rows), sizeof(int));
    ofs.write((const char*)(&out_mat.cols), sizeof(int));
    ofs.write((const char*)(&type), sizeof(int));
    ofs.write((const char*)(out_mat.data), out_mat.elemSize() * out_mat.total());
}

void writeKeyPointBinary(std::ofstream& ofs, const cv::KeyPoint& key_point) {
    ofs.write((const char*)(&key_point.pt.x), sizeof(float));
    ofs.write((const char*)(&key_point.pt.y), sizeof(float));
    ofs.write((const char*)(&key_point.angle), sizeof(float));
    ofs.write((const char*)(&key_point.size), sizeof(float));
    ofs.write((const char*)(&key_point.response), sizeof(float));
    ofs.write((const char*)(&key_point.class_id), sizeof(int));
    ofs.write((const char*)(&key_point.octave), sizeof(int));
}

void writeKeyPointsBinary(std::ofstream& ofs, const std::vector<cv::KeyPoint>& key_point) {
    size_t num_pts = key_point.size();
    ofs.write((const char*)&num_pts, sizeof(size_t));
    for (size_t i = 0; i < num_pts;i++) {
        writeKeyPointBinary(ofs, key_point[i]);
    }
}

int splitStr(const std::string str, std::vector<std::string> *pRet, std::string sep = "/") {
    if (str.empty()) return 0;
    pRet->clear();
    std::string tmp;
    std::string::size_type pos_begin = str.find_first_not_of(sep);
    std::string::size_type comma_pos = 0;
    while (pos_begin != std::string::npos) {
        comma_pos = str.find(sep, pos_begin);
        if (comma_pos != std::string::npos) {
            tmp = str.substr(pos_begin, comma_pos - pos_begin);
            pos_begin = comma_pos + sep.length();
        } else {
            tmp = str.substr(pos_begin);
            pos_begin = comma_pos;
        }
        
        if (!tmp.empty()) {
            (*pRet).push_back(tmp);
            tmp.clear();
        }
    }
    return (int)(*pRet).size();
}

std::vector<std::string> globVector(const std::string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        std::string tm = std::string(glob_result.gl_pathv[i]);
        std::string base_filename = tm.substr(tm.find_last_of("/\\") + 1);
        files.push_back(std::string(base_filename));
    }
    globfree(&glob_result);
    return files;
}

void rootSift(cv::Mat &descriptors, const float eps = 1e-7) {
    // Compute sums for L1 Norm
    cv::Mat sums_vec;
    descriptors = cv::abs(descriptors); //otherwise we draw sqrt of negative vals
    cv::reduce(descriptors, sums_vec, 1 /*sum over columns*/, CV_REDUCE_SUM, CV_32FC1);
    for(int row = 0; row < descriptors.rows; row++) {
        int offset = row*descriptors.cols;
        for(int col = 0; col < descriptors.cols; col++) {
            descriptors.at<float>(offset + col) = sqrt(descriptors.at<float>(offset + col) /
                                                       (sums_vec.at<float>(row) + eps) /*L1-Normalize*/);
        }
        
        // L2 distance
        cv::normalize(descriptors.row(row), descriptors.row(row), 1.0, 0.0, cv::NORM_L2);
    }
    return;
}

int main(int argc, char** argv) {
    
    std::string templatesPath = "/Users/willard/projects/bovw/data";
    std::string modelPath = "/Users/willard/codes/cpp/hnsw_sift_retrieval/hnsw_sifts_retrieval/data/all_1029.dat";
    
    std::vector<std::string> imgPaths = globVector(templatesPath + "/*.*");
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(128);
    std::vector<std::string> imgPathsGood;
    
    for (int i = 0; i < imgPaths.size(); i++) {
        std::string img_name = templatesPath + "/" + imgPaths.at(i);
        cv::Mat im = cv::imread(img_name.c_str(), 1);
        if (im.empty()) continue;
        
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(im, keypoints);
        
        if (keypoints.size() < 5) {
            std::cout << "too few feature points: " << img_name << std::endl;
            continue;
        }
        imgPathsGood.push_back(imgPaths.at(i));
    }
    
    std::ofstream fp;
    fp.open(modelPath.c_str(), std::ios::out|std::ios::binary);
    int templatesNum = (int)imgPathsGood.size();
    fp.write(reinterpret_cast<const char*>(&templatesNum), sizeof(int));
    
    int numTotal = 0;
    
    for (int i = 0; i < templatesNum; i++) {
        std::string img_name = templatesPath + "/" + imgPathsGood.at(i);
        cv::Mat im = cv::imread(img_name.c_str(), 1);
        if (im.empty()) continue;
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat img_descs;
        detector->detect(im, keypoints);
        detector->compute(im, keypoints, img_descs);
        
        if (keypoints.size() < 5) {
            std::cout << "too few feature points: " << img_name << std::endl;
            continue;
        }
        
        printf("%s: %d(%d), %d\n", imgPathsGood.at(i).c_str(), i+1, templatesNum, (int)keypoints.size());
        rootSift(img_descs);
        
        // 获取不带后缀的模板文件名string的大小
        int templateNameSize = (int)imgPathsGood.at(i).size();
        fp.write(reinterpret_cast<const char*>(&templateNameSize), sizeof(int));
        fp.write(reinterpret_cast<const char*>(&numTotal), sizeof(int)); // 增量点数
        fp.write(imgPathsGood.at(i).c_str(), imgPathsGood.at(i).size());
        
        // 模型文件中写入模板图像
        writeMatBinary(fp, img_descs);
        writeKeyPointsBinary(fp, keypoints);
        
        numTotal += keypoints.size();
    }
    
    fp.close();
    
    std::cout << "numTotal: " << numTotal << std::endl;
    return 0;
}
