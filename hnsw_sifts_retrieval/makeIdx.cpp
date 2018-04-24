
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <unordered_set>
#include "hnswlib.h"

using namespace std;
using namespace hnswlib;


/***** test *******/
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <glob.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

typedef unsigned char uint8;


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

void readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
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

void readKeyPointBinary(std::ifstream& ifs, cv::KeyPoint& key_point)
{
    ifs.read((char*)(&key_point.pt.x), sizeof(float));
    ifs.read((char*)(&key_point.pt.y), sizeof(float));
    ifs.read((char*)(&key_point.angle), sizeof(float));
    ifs.read((char*)(&key_point.size), sizeof(float));
    ifs.read((char*)(&key_point.response), sizeof(float));
    ifs.read((char*)(&key_point.class_id), sizeof(int));
    ifs.read((char*)(&key_point.octave), sizeof(int));
}

void readKeyPointsBinary(std::ifstream& ifs, std::vector<cv::KeyPoint>& key_point) {
    size_t num_pts;
    ifs.read((char*)&num_pts, sizeof(size_t));
    key_point.resize(num_pts);
    //key_point.reserve(num_pts);
    for (size_t i = 0; i < num_pts;i++) {
        readKeyPointBinary(ifs, key_point[i]);
    }
}

struct SIFTInfo {
    std::string fileBaseName;
    cv::Mat DESCs;
    std::vector<cv::KeyPoint> KPts;
};

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }
    
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    
    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
    
};


/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;
    
#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);
    
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif
    
#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
    
#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;
    
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
    
#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

// massQA: 1万个查询, 每个查询有1000个最近邻
// massQ: 1万个查询样本记录
// answers: 记录ground truth
//static void get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, L2Space &l2space,
static void get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, InnerProductSpace &l2space,
                   size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    printf("number of queries: %d\n", qsize);
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                         vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        //total += gt.size();
        
        //printf("debug infomation: k: %d, result size: %d, gt size: %d\n", k, result.size(), gt.size());
        
        if(gt.size() != 0){
            g.insert(gt.top().second);
            gt.pop();
        }
        
        while (gt.size()) {
            //g.insert(gt.top().second);
            printf("gt size is not 1\n");
            gt.pop();
        }
        
        total += g.size();
        
        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
        
    }
    return 1.0f * correct / total;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    vector<size_t> efs = {1000};
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();
        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        printf("k: %d \t ef: %d \t recall: %f \t time_us_per_query: %fus\n", k, ef, recall, time_us_per_query);
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

struct geoInfo{
    std::string fileBaseName;
    int ID;
    cv::KeyPoint Pt;
    int relateID;
};

int main(int argc, char** argv) {
    
    std::string modelPath = "/Users/willard/codes/cpp/hnsw_sift_retrieval/hnsw_sifts_retrieval/data/all_1029.dat";
   
    int efConstruction = 80;
    int M = 32;
    size_t vecdim = 128;    // 特征维度
    char path_index[1024];  // 索引文件
    char path_geo[1024];    // 几何信息文件
    
    size_t totalNum = 125402;
    sprintf(path_index, "/Users/willard/codes/cpp/hnsw_sift_retrieval/hnsw_sifts_retrieval/data/sifts_%dm_ef_%d_M_%d_ip.bin", (int)totalNum, efConstruction, M);
    
    sprintf(path_geo, "/Users/willard/codes/cpp/hnsw_sift_retrieval/hnsw_sifts_retrieval/data/sifts_%d_geoInfo.bin", (int)totalNum);
    
    std::vector<geoInfo> geosInfo;
    float *massb = new float[vecdim];

    InnerProductSpace l2space(vecdim);
    HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, totalNum, M, efConstruction);
        
        std::ifstream fp;
        int templatesNum = 0;
        fp.open(modelPath.c_str(), std::ios::in|std::ios::binary);
        fp.read((char*)(&templatesNum), sizeof(int));
        
        int count = 0;
        for (int i = 0; i < templatesNum; i++) {
            int increCount = 0;
            int templateNameSize = 0;
            char templateName[1024] = {""};
            
            cv::Mat desc;
            std::vector<cv::KeyPoint> keyPoints;
            keyPoints.clear();
            
            fp.read((char*)&templateNameSize, sizeof(int));
            fp.read((char*)&increCount, sizeof(int));
            fp.read(templateName, templateNameSize);
            
            readMatBinary(fp, desc);
            readKeyPointsBinary(fp, keyPoints);
            
            for(int j = 0; j  < desc.rows; j++)
            {
                for(int k = 0; k < vecdim; k++)
                {
                    massb[k] = desc.at<float>(j, k);
                }
                geoInfo tmp;
                
                tmp.fileBaseName = templateName;
                tmp.ID = count;
                tmp.Pt = keyPoints.at(j);
                
                tmp.relateID = count - increCount;
                
                geosInfo.push_back(tmp);
                appr_alg->addPoint((void *) (massb), (size_t) count);
                ++count;
                printf("%s: %d\n", templateName, (int)count);
            }
            
            desc.release();
        }
        fp.close();
        
        // 保存几何信息
        std::ofstream fpp;
        fpp.open(path_geo, std::ios::out|std::ios::binary);
        fpp.write((char*)(&count), sizeof(int));
        for(int i = 0; i < count; i++) {
            
            int templateNameSize = (int)geosInfo.at(i).fileBaseName.size();
            fpp.write(reinterpret_cast<const char*>(&templateNameSize), sizeof(int));
            fpp.write(geosInfo.at(i).fileBaseName.c_str(), geosInfo.at(i).fileBaseName.size());
            fpp.write((char*)(&geosInfo.at(i).ID), sizeof(int));
            fpp.write((char*)(&geosInfo.at(i).relateID), sizeof(int));
            
            fpp.write((const char*)(&geosInfo.at(i).Pt.pt.x), sizeof(float));
            fpp.write((const char*)(&geosInfo.at(i).Pt.pt.y), sizeof(float));
            fpp.write((const char*)(&geosInfo.at(i).Pt.angle), sizeof(float));
            fpp.write((const char*)(&geosInfo.at(i).Pt.size), sizeof(float));
            fpp.write((const char*)(&geosInfo.at(i).Pt.response), sizeof(float));
            fpp.write((const char*)(&geosInfo.at(i).Pt.class_id), sizeof(int));
            fpp.write((const char*)(&geosInfo.at(i).Pt.octave), sizeof(int));
        }
        fpp.close();
    }
    
    appr_alg->saveIndex(path_index);
    
    return 0;
}
