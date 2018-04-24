
#include <iostream>
#include <fstream>
#include "vl_covdet.hpp"

using namespace cv;
using namespace std;

typedef unsigned int uint8;

// Get base name from a full path
// Usage: base_name("/xxx/xxx/123.jpg")
// Output: 123.jpg
std::string base_name(std::string const & path, std::string const & delims){
    return path.substr(path.find_last_of(delims) + 1);
}

// Remove extension from a base name
// Usage: remove_extension("123.jpg")
// Output: 123
std::string remove_extension(std::string const & filename){
    typename std::string::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}

// save sift
void exportKeypoints(ostream &out, std::vector<std::vector<float>>& ddesc)
{
    out << 128 << endl;
    out << ddesc.size() << endl;
    for (size_t i=0; i < ddesc.size(); i++)
    {
        out << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0;
        for (size_t j=0; j<128; j++){
            out << " " << uint8(512.0F*ddesc[i][j]);
        }
        out << endl;
    }
}

int main(int argc, char** argv){
	
    // 提取covdet特征
    std::string dimageName = argv[1];
    std::vector<std::vector<float>> dframes;
    std::vector<std::vector<float>> ddesc;
    cv::Mat dMat = cv::imread(dimageName);
    cv::resize(dMat, dMat, cv::Size(int(0.5*dMat.cols), int(0.5*dMat.rows)));
    vl_covdet(dMat, dframes, ddesc);

    char suffix[] = ".covdet.sift";
    string save_dir = argv[2] + remove_extension(base_name(argv[1], "/"));
    int len = strlen(save_dir.c_str()) + strlen(suffix) + 1;
    char buf[len];
    snprintf(buf, len, "%s%s", save_dir.c_str(), suffix); 
    buf[len-1]=0;
    ofstream out(buf);
    exportKeypoints(out, ddesc);
    out.close();
	return 0;
}
