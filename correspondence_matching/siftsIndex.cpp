
#include "siftsIndex.hpp"

void siftsIDX::InitModel(const std::string& geoInfoPath, const std::string& indexInfoPath) {
    
    l2space = new hnswlib::InnerProductSpace(vecdim);
    appr_alg = new hnswlib::HierarchicalNSW<float>(l2space, indexInfoPath, false);
    
    geosInfo.clear();
    if (existsFile(indexInfoPath)) {
        std::cout << "Loading index from " << indexInfoPath << ":\n";
        
        // 读取几何信息
        int count = 0;
        std::ifstream fpp;
        fpp.open(geoInfoPath, std::ios::in|std::ios::binary);
        fpp.read((char*)(&count), sizeof(int));
        
        std::string preName = "";
        
        for(int i = 0; i < count; i++) {
            
            geoInfo tmp;
            
            int templateNameSize = 0;
            int ID = 0;
            int relateID = 0;
            char templateName[1024] = {""};
            
            fpp.read((char*)&templateNameSize, sizeof(int));
            fpp.read(templateName, templateNameSize);
            fpp.read((char*)&ID, sizeof(int));
            fpp.read((char*)&relateID, sizeof(int));
            
            tmp.fileBaseName = templateName;
            tmp.ID = ID;
            tmp.relateID = relateID;
            
            fpp.read((char*)(&tmp.Pt.pt.x), sizeof(float));
            fpp.read((char*)(&tmp.Pt.pt.y), sizeof(float));
            fpp.read((char*)(&tmp.Pt.angle), sizeof(float));
            fpp.read((char*)(&tmp.Pt.size), sizeof(float));
            fpp.read((char*)(&tmp.Pt.response), sizeof(float));
            fpp.read((char*)(&tmp.Pt.class_id), sizeof(int));
            fpp.read((char*)(&tmp.Pt.octave), sizeof(int));
            
            geosInfo.push_back(tmp);
        }
    }
    
    appr_alg->setEf(1000);
}

void siftsIDX::rootSift(cv::Mat &descriptors, const float eps)
{
    // Compute sums for L1 Norm
    cv::Mat sums_vec;
    descriptors = cv::abs(descriptors); //otherwise we draw sqrt of negative vals
    cv::reduce(descriptors, sums_vec, 1 /*sum over columns*/, CV_REDUCE_SUM, CV_32FC1);
    for(int row = 0; row < descriptors.rows; row++){
        int offset = row*descriptors.cols;
        for(int col = 0; col < descriptors.cols; col++){
            descriptors.at<float>(offset + col) = sqrt(descriptors.at<float>(offset + col) /
                                                       (sums_vec.at<float>(row) + eps) /*L1-Normalize*/);
        }
        // L2 distance
        cv::normalize(descriptors.row(row), descriptors.row(row), 1.0, 0.0, cv::NORM_L2);
        
    }
    return;
}

bool siftsIDX::existsFile(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

void siftsIDX::readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
    int rows, cols, type;
    ifs.read((char*)(&rows), sizeof(int));
    if (rows == 0) {
        return;
    }
    ifs.read((char*)(&cols), sizeof(int));
    ifs.read((char*)(&type), sizeof(int));
    
    in_mat.release();
    in_mat.create(rows, cols, type);
    ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());
}

void siftsIDX::readKeyPointBinary(std::ifstream& ifs, cv::KeyPoint& key_point)
{
    ifs.read((char*)(&key_point.pt.x), sizeof(float));
    ifs.read((char*)(&key_point.pt.y), sizeof(float));
    ifs.read((char*)(&key_point.angle), sizeof(float));
    ifs.read((char*)(&key_point.size), sizeof(float));
    ifs.read((char*)(&key_point.response), sizeof(float));
    ifs.read((char*)(&key_point.class_id), sizeof(int));
    ifs.read((char*)(&key_point.octave), sizeof(int));
}

void siftsIDX::readKeyPointsBinary(std::ifstream& ifs, std::vector<cv::KeyPoint>& key_point) {
    size_t num_pts;
    ifs.read((char*)&num_pts, sizeof(size_t));
    key_point.resize(num_pts);
    //key_point.reserve(num_pts);
    for (size_t i = 0; i < num_pts;i++) {
        readKeyPointBinary(ifs, key_point[i]);
    }
}
