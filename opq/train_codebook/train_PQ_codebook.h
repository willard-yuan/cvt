#ifndef TRAIN_PQ_CODEBOOK
#define TRAIN_PQ_CODEBOOK

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

extern "C"{
#include <yael/vector.h>
#include <yael/kmeans.h>
#include <yael/machinedeps.h>
}

using namespace std;

class TrainPQ{
    
public:
    TrainPQ(string modelFile, int maxTrainFeatNum=0, int featDim=128, int coarseK=8192, int pq_k=256, int pq_m=16);
    ~TrainPQ();
    void LoadFeatureSample(string srcDir);
    void IFVPQ();
    void CoarseQuan();
    void ProdQuan();
    void SaveCodebook(string desDir);
    void reorder(float* feat); 
    
private:
    string m_srcDir;
    string m_desDir;
    
    int m_maxTrainFeatNum;
    int m_featNum;
    int m_featDim;
    int m_coarseK;
    int m_pq_k;
    int m_pq_m;
    int m_pq_step;
    float **m_ppFeatData;
    float **m_ppCoarseCluster;
    float **m_ppFeatResidue;
    float ***m_prodQuantizer;

    long int* reorder_;
};

template<typename T> void Delete2DArray(T** &f)
{
    if (f != NULL)
    {
        if (f[0] != NULL)
        {
            delete [] f[0];
            f[0] = NULL;
        }
        delete []f;
        f = NULL;
    }
}


template<typename T> void Init2DArray(T** &f, int row, int col)
{
    T* pf = new T[row * col];
    memset(pf, 0, sizeof(T) * row * col);
    
    f = new T*[row];
    for (int i = 0; i < row; i ++)
    {
        f[i] = pf + i * col;
    }
}


#endif
