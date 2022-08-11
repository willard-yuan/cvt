// 模型文件格式：6 | 123_1 | logo位置信息 | 2(记录数目) | 12(SIFT阈值) | HOG(阈值) | 模板图像

#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <glob.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#define MAX_TEMPLATE_SIZE 1024*12*100

typedef unsigned char uint8;

static std::vector<std::string> logo_template_name_list;

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

// 功能说明：读取配置文件中的参数
int readConfigFile(const std::string &conf, std::map<std::string, uint8> &template_site_code_map, \
                   std::map<std::string, std::vector<std::pair<int, float> > > &constants_threshold_map)
{
    template_site_code_map.clear();
    constants_threshold_map.clear();
    std::ifstream infile;
    infile.open(conf.c_str());
    std::string line;
    int siteflag = 0;
    int consflag = 0;
    while (getline(infile, line)) {
        if (line == "site start") {
            siteflag = 1;
            continue;
        }
        if (line == "site end") {
            siteflag = 0;
            continue;
        }
        if (line == "cons start") {
            consflag = 1;
            continue;
        }
        if (line == "cons end") {
            consflag = 0;
            continue;
        }
        if (siteflag) {
            std::vector<std::string> tmpStr;
            splitStr(line, &tmpStr, " ");
            if (tmpStr.size() != 2) {
                continue;
            }
            std::string lgname = tmpStr[0];
            uint8 site = static_cast<uint8>(atoi(tmpStr[1].c_str()));
            template_site_code_map[lgname] = site;
        }
        if (consflag) {
            std::vector<std::string> tmpStr;
            splitStr(line, &tmpStr, " ");
            if (tmpStr.size() != 3) {
                continue;
            }
            std::string lgname = tmpStr[0];
            int vali_num = atoi(tmpStr[1].c_str());
            float hog_dis = atof(tmpStr[2].c_str());
            constants_threshold_map[lgname].push_back(std::make_pair(vali_num, hog_dis));
        }
    }
    infile.close();
    return 0;
}

// 功能说明：从文件夹中读取文件的路径
// 输入：文件夹的路径
// 输出：文件路径
std::vector<std::string> globVector(const std::string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        std::string tm = std::string(glob_result.gl_pathv[i]);
        std::string base_filename = tm.substr(tm.find_last_of("/\\") + 1);
        files.push_back(std::string(base_filename));
    }
    globfree(&glob_result);
    return files;
}

int main(int argc, char** argv) {
    
    std::string templates_path = "../model/logo_local/";
    std::string conf_path = "../model/logo_local.conf";
    std::string model_path = "../model/logo_local.dat";
    
    std::vector<std::string> imgs_path = globVector(templates_path + "/*.*");
    
    std::map<std::string, uint8> logo_map;
    std::vector<std::string> tmpStr;
    for (int i = 0; i < imgs_path.size(); i++) {
        std::string file_name = imgs_path[i];
        std::cout << "filename: " << file_name << std::endl;
        // 将获取的图片名称如"xxx.jpg"保存在logo_temp_list
        logo_template_name_list.push_back(file_name);
        tmpStr.clear();
        splitStr(file_name, &tmpStr, "_");
        std::string logo_file_name = tmpStr[0];
        logo_map[logo_file_name] = 1;
    }
    
    std::map<std::string, uint8> logo_site_code_map; // <logo_file_name, site_code>, 不带jpg后缀
    std::map<std::string, std::vector<std::pair<int, float> > > constants_threshold_map; // <logo_file_name, <match_points, hog_distance>>
    // 读取配置文件并将值保存在相应的变量中
    readConfigFile(conf_path, logo_site_code_map, constants_threshold_map);
    
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    
    // 获取模板文件数量，注意是从模板夹中获取的文件模板数目，非常重要
    int logo_img_num = (int)logo_template_name_list.size();
    std::cout << "logo name num: " << logo_map.size() << std::endl; // logo类型数目
    std::cout << "logo temp num: " << logo_template_name_list.size() << std::endl; // 模板数目
    
    // 打开文件，写入模板总数目
    FILE* fp = fopen(model_path.c_str(), "wb");
    fwrite(&logo_img_num, sizeof(int), 1, fp);
    // 设置缓冲区域
    int max_size = MAX_TEMPLATE_SIZE;
    unsigned char* img_buf = new unsigned char[max_size];
    
    for (int i = 0; i < logo_template_name_list.size(); i++)
    {
        // 读入模板文件
        std::string img_name = templates_path + "/" + logo_template_name_list[i];
        cv::Mat im = cv::imread(img_name.c_str(), 1);
        /*cv::imshow("test", im);
         cv::waitKey(0);*/
        if (im.empty()) continue;
        
        // 关键点检测
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(im, keypoints);
        std::cout << logo_template_name_list[i] << ", key points number: " << keypoints.size() << std::endl;
        
        // 关键点数目小于5，该模板忽略掉
        if (keypoints.size() < 5) {
            std::cout << "too few feature points: " << img_name << std::endl;
            continue;
        }
        
        // 读入模板图像文件，获取文件大小
        FILE* fg = fopen(img_name.c_str(), "rb");
        fseek(fg, 0, SEEK_END);
        int img_size = (int)ftell(fg);
        fseek(fg, 0, SEEK_SET);
        std::cout << "template image size: " << img_size << std::endl;
        
        // 模板图像过大忽略模板
        /*if (img_size > max_size) {
            std::cout << "image size too large: " << img_name << std::endl;
            fclose(fg);
            continue;
        }*/
        std::vector<std::string> tmpStr;
        
        // 分割file_name和后缀
        splitStr(logo_template_name_list[i], &tmpStr, "."); // <[0] name, [1] jpg>
        std::string file_name = tmpStr[0];
        splitStr(file_name, &tmpStr, "_");
        std::string logo_name = tmpStr[0]; // 模板类型
        
        // 获取不带后缀的模板文件名string的大小
        int name_size = (int)file_name.size();
        
        // 模型文件中写入模板文件名string的大小以及不带后缀的文件名
        fwrite(&name_size, sizeof(int), 1, fp);
        fwrite(file_name.c_str(), sizeof(char), name_size, fp);
        
        uint8 site = 0;
        // logo模板文件名并不是都是"xxx_x.jpg"形式的，有可能有"xxx.jpg"形式
        // logo_name的位置信息优先
        auto itl = logo_site_code_map.find(logo_name);
        if (itl == logo_site_code_map.end()) {
            auto itt = logo_site_code_map.find(file_name);
            if (itt == logo_site_code_map.end()) {
                // 如果该logo没有设置位置信息，则模板位置信息直接设置为0
                site = 0;
            } else {
                // 否则模板位置信息设置为配置文件中的位置信息
                site = itt->second;
            }
        } else {
            uint8 s0 = itl->second;
            auto itt = logo_site_code_map.find(file_name);
            if (itt == logo_site_code_map.end()) {
                // 有logo_name位置信息，但无模板位置信息，使用logo_name的位置信息
                site = s0;
            } else {
                // 有logo_name位置信息，也有模板位置信息
                // s0为logo_name位置信息，s1为模板位置信息
                uint8 s1 = itt->second;
                site = 0;
                // 对于这种情况，位置信息去1、3、7、15、31
                for (int k = 0; k < 5; k++) {
                    if (s0 % 2 || s1 % 2) site += pow(2, k);
                    s0 /= 2;
                    s1 /= 2;
                }
            }
        }
        // 模型文件写入位置信息
        fwrite(&site, sizeof(uint8), 1, fp);
        
        std::vector<std::pair<int, float> > constants;
        auto itconsl = constants_threshold_map.find(logo_name);
        if (itconsl == constants_threshold_map.end()) {
            auto itconst = constants_threshold_map.find(file_name);
            // logo_name阈值信息不存在，模板阈值信息存在，则使用模板阈值信息
            if (itconst != constants_threshold_map.end()) constants = itconst->second;
        } else {
            // logo_name阈值信息存在
            constants = itconsl->second;
            auto itconst = constants_threshold_map.find(file_name);
            if (itconst != constants_threshold_map.end()) {
                // logo_name阈值信息存在，模板阈值信息也存在，则优先使用logo_name的阈值信息
                // 这里需要改进：优先使用模板位置信息
                for (int k = 0; k < itconst->second.size(); k++) {
                    constants.push_back(itconst->second[k]);
                }
            }
        }
        
#if 0
        std::cout << "name: " << tempname << std::endl;
        std::cout << "site: " << (int)site << std::endl;
        for (int k = 0; k < cons.size(); k++) {
            std::cout << cons[k].first << " " << cons[k].second << std::endl;
        }
#endif
        // 阈值记录条数，模型文件中写入阈值记录条数，模型文件中逐条写入阈值记录信息
        int cons_num = (int)constants.size();
        fwrite(&cons_num, sizeof(int), 1, fp);
        for (int k = 0; k < constants.size(); k++) {
            int vali_num = constants[k].first;
            float hog_dis = constants[k].second;
            fwrite(&vali_num, sizeof(int), 1, fp);
            fwrite(&hog_dis, sizeof(float), 1, fp);
        }
        
        // 模型文件中写入模板图像
        fread(img_buf, sizeof(char), img_size, fg);
        fwrite(&img_size, sizeof(int), 1, fp);
        fwrite(img_buf, sizeof(char), img_size, fp);
        fclose(fg);
    }
    
    delete [] img_buf;
    fclose(fp);
    std::cout << "complete" << std::endl;
    return 0;
}
