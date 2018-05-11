#ifndef IVFOPQ_SEARCH_H
#define IVFOPQ_SEARCH_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <algorithm>
#include <queue>

typedef unsigned char uchar;

using namespace std;
const int max_path = 260;
struct ImgNameStruct
{
    char ptr[max_path];
};

struct IVFelem
{
    int videoId;
    string basename;
    uchar PQindex[16];
};

class IVFOPQ
{
    
public:
    IVFOPQ();
    IVFOPQ(int maxIndexNum);
    ~IVFOPQ();
    int LoadModel(std::string modelFile); // 载入模型
    void LoadIndex(string srcFile);
    void SaveIndex(string desDir);
    void Add(float** m_ppFeat, const int m_frameNum);
    void IndexDatabase(vector<string> featFiles);
    void Query(string featFile, vector<vector<float> >& matchScore, int nk = 3);
    void QueryThrehold(string featFile, vector<vector<float> >& matchScore, int nk = 3);
    void LoadSingleFeatFile(string srcFile, float**& m_ppFeat, int& m_frameNum);
    
    ImgNameStruct* m_imgLocation;
    
private:
    void init();
    void reorder(float* feat);
    
private:
    float** m_ppCoarseCluster;
    float*** m_prodQuantizer;
    int* m_ivfSize;
    int * reorder_;
    
    vector<vector<IVFelem> > m_ivfList;
    
    int m_coarseK;
    int m_pq_m; // 子段维度
    int m_pq_k; // 子段聚类数目
    int m_pq_step;
    int m_featDim;
    int m_imgNum;
    int m_maxIndexNum;  // 最大索引图片数目
    std::string basename;
};
#endif
