
#include "train_PQ_codebook.h"

using namespace std;

TrainPQ::TrainPQ(string modelFile, int maxTrainFeatNum, int featDim, int coarseK, int pq_k, int pq_m):m_maxTrainFeatNum(maxTrainFeatNum),m_featDim(featDim),m_coarseK(coarseK),m_pq_k(pq_k),m_pq_m(pq_m)
{
    m_pq_step = m_featDim/m_pq_m;
    m_ppFeatData = NULL;
    m_ppCoarseCluster = NULL;
    m_prodQuantizer = NULL;
    m_ppFeatResidue = NULL;

    // 读入reorder文件
    printf("feature dimension: %d\n", featDim);
    ifstream fin(modelFile, ios::binary); 
    reorder_ = new long int[m_featDim];
    fin.read((char*)reorder_, sizeof(long int)*m_featDim);
    fin.close();
    for(int i = 0; i < m_featDim; i++){
        std::cout << reorder_[i] << " ";
    }
    std::cout << std::endl;
}

TrainPQ::~TrainPQ()
{
    Delete2DArray(m_ppFeatData);
    Delete2DArray(m_ppCoarseCluster);
    Delete2DArray(m_ppFeatResidue);
    
    if(m_prodQuantizer != NULL)
    {
        for(int i = 0; i < m_pq_m; i++)
        {
            Delete2DArray(m_prodQuantizer[i]);
        }
        delete []m_prodQuantizer;
        m_prodQuantizer = NULL;
    }
    
    delete [] reorder_;
    reorder_ = NULL; 
}

void TrainPQ::LoadFeatureSample(string srcDir)
{
    m_srcDir = srcDir;
    FILE* pFile = fopen(m_srcDir.c_str(), "rb");
    if (pFile == NULL)
    {
        cout << "Fail to open the source file " << srcDir << endl;
        return;
    }
    else
    {
        cout << "Load training features, please wait...\n";
        fseek(pFile, 0, SEEK_END);
        long long nFileLen = ftell(pFile);
        fseek(pFile, 0, SEEK_SET);
        int f_size = m_featDim*sizeof(float);
        int num = int(nFileLen / f_size);
        m_featNum = min(num, m_maxTrainFeatNum); // 训练特征数目
        
        
        Init2DArray(m_ppFeatData, m_featNum, m_featDim);
        int numR = 200000;
        float** ppBuffer = NULL;
        Init2DArray(ppBuffer, numR, m_featDim);
        int nRead;
        
        if (m_featNum < numR)
        {
            nRead = (int)fread(ppBuffer[0], sizeof(float), m_featDim*m_featNum, pFile);
            for (int m = 0; m < m_featNum; m++)
            {
                for (int n = 0; n < m_featDim; n++)
                {
                    //m_ppFeatData[m][n] = float(ppBuffer[m][n]);
                    m_ppFeatData[m][n] = float(ppBuffer[m][reorder_[n]]);
                }
            }
            
        }
        else
        {
            int readTimes = m_featNum / numR;
            int inc = 0;
            for (int i = 0; i < readTimes; i++)
            {
                nRead = (int)fread(ppBuffer[0], sizeof(float),m_featDim*numR, pFile);
                
                for (int m = 0; m < numR; m++)
                {
                    for (int n = 0; n < m_featDim; n++)
                    {
                        //m_ppFeatData[m + inc][n] = float(ppBuffer[m][n]);
                        m_ppFeatData[m + inc][n] = float(ppBuffer[m][reorder_[n]]);
                    }
                }
                inc += numR;
            }
            int tmp = m_featNum - readTimes * numR;
            if (tmp > 0)
            {
                nRead = (int)fread(ppBuffer[0], sizeof(float),m_featDim*tmp, pFile);
                for (int m = 0; m < tmp; m++)
                {
                    for (int n = 0; n < m_featDim; n++)
                    {
                        //m_ppFeatData[m + inc][n] = float(ppBuffer[m][n]);
                        m_ppFeatData[m + inc][n] = float(ppBuffer[m][reorder_[n]]);
                    }
                }
                inc += tmp;
                assert(inc == m_featNum);
            }
        }

        fclose(pFile);
        Delete2DArray(ppBuffer);
        cout << m_featNum << " training features loaded!\n";
    }
}

void TrainPQ::reorder(float* feat)
{
    if(feat == NULL){
        return;
    }
    
    float * temp = new float [m_featDim];
    for (int i=0; i < m_featDim; i++)
        temp[i] = feat[reorder_[i]];
    
    for (int i=0; i< m_featDim; i++)
    {
        feat[i] = temp[i];
    }
    delete [] temp;
}

void TrainPQ::IFVPQ()
{
    CoarseQuan();
    ProdQuan();
    
}

void TrainPQ::CoarseQuan()
{
    Init2DArray(m_ppCoarseCluster, m_coarseK, m_featDim);
    int* pAssignID = new int[m_featNum];
    float* pDists = new float[m_featNum];
    int* pClusSize = new int[m_coarseK];
    
    int numThread = 64;
    int niter = 0;//0 for convergence
    int redo = 1;
    
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    double timeuse;
    kmeans(m_featDim, m_featNum, m_coarseK, niter, m_ppFeatData[0], numThread, 1, redo, m_ppCoarseCluster[0], pDists, pAssignID, pClusSize);
    
    gettimeofday(&endtime, NULL);
    timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
    timeuse/=1000;
    cout<<"Coarse quantion time: "<<timeuse<<" ms"<<endl;
    
    /*-----------------------------check result--------------------------*/
    
    for(int i=0;i<m_coarseK;i++)
    {
        if(pClusSize[i]<=0)
        {
            cout<<"warning: empty cluster 1: "<<i<<endl;
        }
    }
    
    
    /*-------------------------------------------------------------------*/
    
    Init2DArray(m_ppFeatResidue, m_featNum, m_featDim);
    for(int i = 0; i < m_featNum; i++)
    {
        for(int j = 0; j < m_featDim; j++)
        {
            m_ppFeatResidue[i][j] = m_ppFeatData[i][j] - m_ppCoarseCluster[pAssignID[i]][j];
        }
    }
    cout<<"finish coarse quantization."<<endl;
    delete[] pAssignID;
    delete[] pDists;
    delete[] pClusSize;
    
}

void TrainPQ::ProdQuan()
{
    cout<<"product quantization......."<<endl;
    m_prodQuantizer = new float**[m_pq_m];
    for(int i = 0; i < m_pq_m; i++)
    {
        Init2DArray(m_prodQuantizer[i], m_pq_k, m_pq_step);
    }
    
#pragma omp parallel for
    for(int i = 0; i < m_pq_m; i++)
    {
        float* v = new float[m_featNum*m_pq_step];
        cout<<m_pq_step<<endl;
        for(int j=0; j<m_featNum; j++)
        {
            memcpy(v+j*m_pq_step, m_ppFeatResidue[0]+j*m_featDim+i*m_pq_step, sizeof(float)*m_pq_step);
        }
        cout<<"2"<<endl;
        int* pAssignID = new int[m_featNum];
        float* pDists = new float[m_featNum];
        int* pClusSize = new int[m_pq_k];
        int numThread = 1;
        int niter = 0;//0 for convergence
        int redo = 1;
        
        struct timeval starttime, endtime;
        gettimeofday(&starttime, NULL);
        double timeuse;
        
        kmeans(m_pq_step, m_featNum, m_pq_k, niter, v, numThread, 1, redo, m_prodQuantizer[i][0], pDists, pAssignID, pClusSize);
        
        gettimeofday(&endtime, NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
        timeuse/=1000;
        cout<<"product quantion "<< i <<"      time: "<<timeuse<<" ms"<<endl;
        
        delete[] v;
        delete[] pAssignID;
        delete[] pDists;
        delete[] pClusSize;
        
    }
    
    
}


void TrainPQ::SaveCodebook(string desDir)
{
    string strFeatNum, strK, strDim, str_pq_m, str_pq_k;
    stringstream sin_str;
    sin_str << m_featNum;
    sin_str >> strFeatNum;
    sin_str.clear();
    
    sin_str << m_coarseK;
    sin_str >> strK;
    sin_str.clear();
    
    sin_str << m_featDim;
    sin_str >> strDim;
    sin_str.clear();
    
    sin_str << m_pq_m;
    sin_str >> str_pq_m;
    sin_str.clear();
    
    sin_str << m_pq_k;
    sin_str >> str_pq_k;
    sin_str.clear();
    
    m_desDir = desDir + "/OPQ_db_" + strFeatNum +"_dim_" + strDim + "_k_" + strK +"_PQ_m"+str_pq_m + "_k"+str_pq_k+ ".fvecs";
    
    ofstream outFile(m_desDir.c_str(), ios::binary);
    
    outFile.write((char*)&m_featDim, sizeof(int));
    outFile.write((char*)&m_coarseK, sizeof(int));
    outFile.write((char*)&m_pq_m, sizeof(int));
    outFile.write((char*)&m_pq_k, sizeof(int));
    outFile.write((char*)m_ppCoarseCluster[0], sizeof(float) * m_featDim * m_coarseK);
    
    for(int i=0;i<m_pq_m;i++)
    {
        outFile.write((char*)m_prodQuantizer[i][0], sizeof(float) * m_pq_step * m_pq_k);
    }
     
    outFile.write((char*)reorder_, sizeof(int) * m_featDim);
    outFile.close();
}
