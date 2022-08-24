#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <glob.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "../include/logo.h"

using namespace std;

#define TIME_DIFF(start, end) ((end.tv_sec-start.tv_sec)*1000 + (end.tv_nsec-start.tv_nsec)/1000000.0)

int splitStr(const std::string str, std::vector<std::string> *pRet, std::string sep = "/") {
    if (str.empty()) {
        return 0;
    }
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

int LoadFileContent(std::string& filename, std::vector<char>* content) {
    std::ifstream ifs(filename.c_str());
    content->clear();
    content->assign((std::istreambuf_iterator<char>(ifs)),
                    (std::istreambuf_iterator<char>()));
    if (content->size() <= 0) {
        std::cout << "Failed in opening & reading file:\t" << filename << std::endl;
        return -1;
    }
    return 0;
}

// 由路径获取文件名
vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        auto tmp = string(glob_result.gl_pathv[i]);
        files.push_back(tmp);
    }
    std::sort(files.begin(), files.end());
    globfree(&glob_result);
    return files;
}

int main(int argc, const char * argv[]) {

    // images to be detected
    vector<string> FLAGS_ims = globVector("../data/*");
    std::string FLAGS_dat = "../model/logo_local.dat";
    struct timespec t0, t1;
    cvt::Logo* logo = cvt::LogoFactory::Create();
    std::cout << "FLAGS_dat: " << FLAGS_dat << std::endl;

    logo->Init(FLAGS_dat);

    for (int i = 0; i < FLAGS_ims.size(); i++){
        std::vector<std::string> tmpStr;
        splitStr(FLAGS_ims[i], &tmpStr, "/");
        std::string frame_id = tmpStr[tmpStr.size()-1];
        splitStr(frame_id, &tmpStr, ".");
        frame_id = tmpStr[0];
        std::string res = "";
        clock_gettime(CLOCK_MONOTONIC, &t0);
        logo->LogoDetect(FLAGS_ims[i], frame_id, &res);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (res == "") {
            res = "nologo";
#if 0
            cout << FLAGS_ims[i] + "\n";
#endif
        }
        if (frame_id != res) std::cout << frame_id << ", detect result: " << res << std::endl;
    }
    printf("logo detect time: %fms\n", TIME_DIFF(t0, t1));
    return 0;
}
