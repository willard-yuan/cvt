#include "base_utils.hpp"
#include "opencv_utils.hpp"

using namespace std;
using namespace cv;

template<typename Out>
void split(const std::string &s, char delim, Out result, std::string &frame_id) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, delim)) {
        if (i == 0){
            frame_id = item;
        }
        if (i >= 1){
            *(result++) = std::stof(item);
        }
        ++i;
    }
}

std::vector<float> split(const std::string &s, char delim, std::string &frame_id) {
    std::vector<float> elems;
    split(s, delim, std::back_inserter(elems), frame_id);
    return elems;
}

int main(int argc, char** argv) {
    
    // 读取特征
    std::string line;
    std::string frame_id;
    //std::string feats_path = argv[1];
    std::string feats_path = "data.txt";
    std::ifstream fin(feats_path);
    std::vector<std::vector<float>> feats;
    int i = 0;
    while (std::getline(fin, line)){
        std::vector<float> data = split(line, ',', frame_id);
        if(data.size() != 2048) {
            std::cout << "feat size != 2048 " << std::endl;
            continue;
        }
        feats.push_back(data);
        ++i;
        if (i%10000 == 0){
            std::cout << "---> " << i << std::endl;
        }
    }
    
    // PCA设置参数
    string filename = "pca.yml";
    int num_reduced_dim = 256;
    
    // 模型训练
    cv::Mat feats_mat = convert2Mat(feats);
    trainPCA(feats_mat, num_reduced_dim, filename);
    
    return 0;
}


#if 0
// small demo
int main(int argc, char** argv) {
    
    string filename = "pca.yml";
    int num_reduced_dim = 4;
    
    // 总共6个数据，每个数据只有4维
    Mat data(6, 4, CV_32FC1);
    float dt[] = {1,2,7,4,
        1,3,6,5,
        1,2,3,4,
        1,3,8,2,
        2,4,6,5,
        3,2,2,4};
    data.data = (uchar*)dt;
    
    trainPCA(data, num_reduced_dim, filename);
    cv::PCA pca = loadPCA(filename);
    
    // 降维方式1
    Mat point = pca.project(data.row(0));
    
    // 降维方式2
    cv::Mat Z = data.row(0) - pca.mean;
    cv::Mat res = Z*pca.eigenvectors.t();
    
    // 验证恢复结果
    Mat result = pca.backProject(point);
    
    cout << "原始数据:\n" << data << endl;
    cout << "pca恢复结果:\n" << result << endl;
    cout << "pca降维结果 :\n" << point << endl;
    cout << "相乘结果：\n" << res << endl;
    
    return 0;
}
#endif
