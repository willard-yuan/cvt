#ifndef baseUUtils_h
#define baseUUtils_h

#include <stdio.h>
#include <glob.h>
#include <math.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

/**********************************************
 * File processing functions
 **********************************************/

// Split string by special char, such as by space
// Usage: split("hello world", ' ')
inline std::vector<std::string> splitString(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline int splitStr(const std::string str, std::vector<std::string> *pRet, std::string sep = "/") {
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

// Split string by substring
inline void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c){
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
        
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

// Get the files full paths from directory
inline std::vector<std::string> globVector(const std::string& pattern)
{
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        auto tmp = std::string(glob_result.gl_pathv[i]);
        files.push_back(tmp);
    }
    std::sort(files.begin(), files.end());
    globfree(&glob_result);
    return files;
}

// Get base name from a full path
// Usage: base_name("/xxx/xxx/123.jpg")
// Output: 123.jpg
inline std::string getBaseName(std::string const & path, std::string const & delims)
{
    return path.substr(path.find_last_of(delims) + 1);
}

// Remove extension from a base name
// Usage: remove_extension("123.jpg")
// Output: 123
inline std::string removeExtension(std::string const & filename){
    typename std::string::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}


// convert file to memory
void load_file(const std::string &filename, std::vector<unsigned char> &buf)
{
    std::ifstream ifs(filename, std::ifstream::in);
    if(ifs.is_open())
    {
        ifs.seekg(0, std::ios::end);
        int length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        buf.resize(length);
        ifs.read((char*)buf.data(), length);
    }
}


/**********************************************
 * Math processing functions
 **********************************************/

// Compute cosine similarity between two vectors
inline float cosine_similarity(std::vector<float> &v1, std::vector<float> &v2){
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    dot = std::inner_product( v1.begin(), v1.end(), v2.begin(), 0.0 );
    denom_a = std::inner_product( v1.begin(), v1.end(), v1.begin(), 0.0 );
    denom_b = std::inner_product( v2.begin(), v2.end(), v2.begin(), 0.0 );
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

// Normalize vector by L2 norm
inline int nomalizeVector(std::vector<float> &v, const int feature_dim)
{
    if (v.size() !=feature_dim) return 0;
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++){
        *it = (*it)/denorm_v;
    }
    return 1;
}

inline float distanceL2(const std::vector<float> &x, const std::vector<float> &y) {
    if (x.size() != y.size()) return -1;
    float dis = 0;
    for (int i = 0; i < x.size(); i++) {
        float dif = x[i] - y[i];
        dis += dif*dif;
    }
    return sqrt(dis);
}

#endif
