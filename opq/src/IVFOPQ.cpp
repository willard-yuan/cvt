
#include "common.h"
#include "IVFOPQ.h"

const float threhold = 1.0;


void IVFOPQ::init()
{
    m_ppCoarseCluster = NULL;
    m_prodQuantizer = NULL;
    m_imgLocation = NULL;
    m_ivfSize = NULL;
    m_imgNum = 0;
}

IVFOPQ::IVFOPQ(int maxIndexNum):m_maxIndexNum(maxIndexNum)
{
    init();
}

IVFOPQ::IVFOPQ()
{
    init();
}

IVFOPQ::~IVFOPQ()
{
    Delete2DArray(m_ppCoarseCluster);
    //	Delete2DArray(m_ppFeat);
    
    if(m_ivfSize != NULL)
    {
        delete[] m_ivfSize;
        m_ivfSize = NULL;
    }
    
    if(m_imgLocation != NULL)
    {
        delete[] m_imgLocation;
        m_imgLocation = NULL;
    }
    
    if(m_prodQuantizer != NULL)
    {
        for(int i = 0; i < m_pq_m; i++)
        {
            Delete2DArray(m_prodQuantizer[i]);
        }
        delete []m_prodQuantizer;
        m_prodQuantizer = NULL;
    }

}

// 载入OPQ模型
// m_featDim: 特征维度，128
// m_coarseK: 一级聚类中心数目，8192
// m_pq_m: 子段数目，16
// m_pq_k: 二级聚类中心数目，256
// m_ppCoarseCluster: 一级聚类中心, 8192*128
// m_prodQuantizer: 二级聚类中心，16*(256*8)
// reorder_: 变换顺序
int IVFOPQ::LoadModel(std::string modelFile)
{
    printf("load training file....\n");
    
    ifstream fin(modelFile, ios::binary);
    if(!fin.is_open())
    {
        printf("Can not open the model file!\n");
        return 0;
    }
    
    fin.read((char*)&m_featDim, sizeof(int));
    fin.read((char*)&m_coarseK, sizeof(int));
    fin.read((char*)&m_pq_m, sizeof(int));
    fin.read((char*)&m_pq_k, sizeof(int));
    m_pq_step = m_featDim / m_pq_m;
    
    printf("feature dimension: %d, coarse codebook size: %d, subspace dimension: %d, number of subspace codebook: %d, finetune codebook size: %d\n", m_featDim, m_coarseK, m_pq_step, m_pq_m, m_pq_k);
    
    // 读入一级量化器
    Init2DArray(m_ppCoarseCluster, m_coarseK, m_featDim);
    fin.read((char*)m_ppCoarseCluster[0], sizeof(float)*m_coarseK*m_featDim);
    
    // 读入二级量化器
    m_prodQuantizer = new float**[m_pq_m];
    for(int i=0; i<m_pq_m; i++)
    {
        Init2DArray(m_prodQuantizer[i], m_pq_k, m_pq_step);
        fin.read((char*)m_prodQuantizer[i][0], sizeof(float)*m_pq_k*m_pq_step);
    }
    reorder_ = new int[m_featDim];
    fin.read((char*)reorder_, sizeof(int)*m_featDim);
    fin.close();
    
    // 初始化倒排索引的拉链长度
    m_ivfList.resize(m_coarseK);
    
    return 1;
}

// 添加进索引
void IVFOPQ::Add(float** m_ppFeat, const int m_frameNum)
{
    for(int f = 0; f < m_frameNum; f++)
    {
        float tmp;
        int vw = -1;
        float dismin = UINT_MAX;
        
        // 寻找一级量化器距离最近的类中心
        for(int i = 0; i < m_coarseK; i++)
        {
            
            float distmp = 0.0f;
            for(int j = 0; j < m_featDim; j++)
            {
                tmp = m_ppFeat[f][j] - m_ppCoarseCluster[i][j];
                distmp += tmp*tmp;
            }
            
            if(distmp < dismin)
            {
                dismin = distmp;
                vw = i; // 一级量化后所在的位置（编码）
            }
        }
        
        IVFelem elem;
        elem.videoId = m_imgNum; // videoId记录的是编号
        
        // 计算一级码本的量化残差
        float* feat_residual = new float[m_featDim];
        for(int i = 0; i < m_featDim; i++)
        {
            feat_residual[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
        }
        
        for(int i = 0; i < m_pq_m; i++)
        {
            float dismin1 = UINT_MAX;
            int vw1 = -1;
            for(int j = 0; j < m_pq_k; j++)
            {
                float distmp1 = 0.0f;
                float tmp;
                // 计算二级码本的量化残差
                for(int k = 0; k < m_pq_step; k++)
                {
                    tmp = feat_residual[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
                    distmp1 += tmp*tmp;
                }
                if(distmp1 < dismin1)
                {
                    dismin1 = distmp1;
                    vw1 = j; // 二级量化后所在的位置（编码）
                }
            }
            elem.PQindex[i] = vw1;
            printf("%d ", vw1);
        }
        
        printf("\n");
        
        m_ivfList.at(vw).push_back(elem);
        delete[] feat_residual;
        feat_residual = NULL;
    }
    
    printf("\n");
    
}

void IVFOPQ::IndexDatabase(vector<string> featFiles)
{
    int num = min(int(featFiles.size()), m_maxIndexNum);
    m_imgLocation = new ImgNameStruct[num];
    
    for(int i = 0; i < num; i++)
    {
        std::cout << featFiles.at(i) << std::endl;
        
        struct timeval starttime, endtime;
        double timeuse;
        gettimeofday(&starttime, NULL);
        
        float** m_ppFeat = NULL;
        Delete2DArray(m_ppFeat);
        
        int m_frameNum = 0;
        std::string srcFile = featFiles.at(i);
        LoadSingleFeatFile(srcFile, m_ppFeat, m_frameNum);
        
        if(m_frameNum > 0)
        {
            Add(m_ppFeat, m_frameNum);
            memset(m_imgLocation[m_imgNum].ptr, 0, max_path * sizeof(char));
            strcpy(m_imgLocation[m_imgNum].ptr, srcFile.c_str());
            m_imgNum++;
        }
        gettimeofday(&endtime, NULL);
        timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + (endtime.tv_usec - starttime.tv_usec);
        timeuse /= 1000;
        
        //cout << "Id:" << i << "    timeuse: " << timeuse << endl;
        
    }
    
}

void IVFOPQ::Query(string featFile, vector<vector<float> >& matchScore, int nk)
{
    //	Delete2DArray(m_ppFeat);
    //	m_frameNum = 0;
    float** m_ppFeat;
    
    int m_frameNum = 0;
    LoadSingleFeatFile(featFile, m_ppFeat, m_frameNum);
    if(m_frameNum == 0)
    {
        return;
    }
    
    matchScore.resize(m_frameNum);
    
    priority_queue<pair<float, int> > qu;
    
    int aver_cand = 0;
    
    for(int f = 0; f < m_frameNum; f++)
    {
        qu = priority_queue<pair<float, int> >(); // clear priority_queue
        float tmp;

        // 寻找倒排链
        for(int i = 0; i < m_coarseK; i++)
        {
            float dis_tmp = 0.0f;
            for(int j = 0; j < m_featDim; j++)
            {
                
                tmp = (m_ppFeat[f][j] - m_ppCoarseCluster[i][j]);
                dis_tmp += tmp*tmp;
            }
            
            if(i < nk)
            {
                qu.push({dis_tmp,i});
            }
            else
            {
                if(dis_tmp < qu.top().first)
                {
                    qu.pop();
                    qu.push({dis_tmp,i});
                }
            }
        }
        
        matchScore.at(f).resize(m_imgNum, threhold);
        
        for(int nq_id = 0; nq_id < nk; nq_id++)
        {
            int vw = qu.top().second;
            qu.pop();
            
            float* feat_res = new float[m_featDim];
            float** PQ_table;
            Init2DArray(PQ_table, m_pq_m, m_pq_k);
            // 查询向量的残差
            for(int i = 0; i < m_featDim; i++)
            {
                feat_res[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
            }
            // m_pq_m: 子段数目
            // m_pq_k: 子段类中心数目
            for(int i = 0; i < m_pq_m; i++)
            {
                for(int j = 0; j < m_pq_k; j++)
                {
                    float dis = 0.0f;
                    for(int k = 0; k < m_pq_step; k++)
                    {
                        tmp = feat_res[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
                        dis += tmp*tmp;
                    }
                    PQ_table[i][j] = dis;
                }
            }
            
            printf("\n====================PQ table begin ====================\n");
            print2DArray(PQ_table, m_pq_m, m_pq_k);
            printf("\n====================PQ table end ====================\n");
            
            
            aver_cand = + (int)m_ivfList[vw].size();
            
            for(int j = 0; j < m_ivfSize[vw]; j++)
            {
                float score = 0.0f;
                for(int k = 0; k < m_pq_m; k++)
                {
                    score += PQ_table[k][m_ivfList[vw][j].PQindex[k]];
                }
                std::cout << "score: " << score << std::endl;
                matchScore.at(f).at(m_ivfList[vw][j].videoId) = min(score, matchScore.at(f).at(m_ivfList[vw][j].videoId));
            }
            delete[] feat_res;
            Delete2DArray(PQ_table);
            
        }
        
    }
    
    printf("average number of candidate frames: %f\n", aver_cand/(1.0*m_frameNum));
    
    Delete2DArray(m_ppFeat);
}

void IVFOPQ::QueryThrehold(string featFile, vector<vector<float> >& matchScore, int nk)
{
    //	Delete2DArray(m_ppFeat);
    //	m_frameNum = 0;
    float** m_ppFeat;
    
    int m_frameNum = 0;
    LoadSingleFeatFile(featFile, m_ppFeat, m_frameNum);
    if(m_frameNum == 0)
    {
        return;
    }
    
    matchScore.resize(m_frameNum);
    
    priority_queue<pair<float, int> > qu;
    
    int aver_cand = 0;
    
    for(int f = 0; f < m_frameNum; f++)
    {
        qu = priority_queue<pair<float, int> >(); // clear priority_queue
        float tmp;
        // 对于每帧
        for(int i = 0; i < m_coarseK; i++)
        {
            float dis_tmp = 0.0f;
            for(int j = 0; j < m_featDim; j++)
            {
                
                tmp = (m_ppFeat[f][j] - m_ppCoarseCluster[i][j]);
                dis_tmp += tmp*tmp;
            }
            
            if(i < nk)
            {
                qu.push({dis_tmp,i});
            }
            else
            {
                if(dis_tmp < qu.top().first)
                {
                    qu.pop();
                    qu.push({dis_tmp,i});
                }
            }
        }
        matchScore.at(f).resize(m_imgNum, threhold);
        
        for(int nq_id = 0; nq_id < nk; nq_id++)
        {
            int vw = qu.top().second;
            qu.pop();
            
            float* feat_res = new float[m_featDim];
            float** PQ_table;
            Init2DArray(PQ_table, m_pq_m, m_pq_k);
            // 查询向量的残差
            for(int i = 0; i < m_featDim; i++)
            {
                feat_res[i] = m_ppFeat[f][i] - m_ppCoarseCluster[vw][i];
            }
            // m_pq_m: 子段数目
            // m_pq_k: 子段类中心数目
            for(int i = 0; i < m_pq_m; i++)
            {
                for(int j = 0; j < m_pq_k; j++)
                {
                    float dis = 0.0f;
                    for(int k = 0; k < m_pq_step; k++)
                    {
                        tmp = feat_res[i*m_pq_step+k] - m_prodQuantizer[i][j][k];
                        dis += tmp*tmp;
                    }
                    PQ_table[i][j] = dis;
                }
            }
            
            //print2DArray(PQ_table, m_pq_m, m_pq_k);
            aver_cand = + (int)m_ivfList[vw].size();
            
            for(int j = 0; j < m_ivfSize[vw]; j++)
            {
                float score = 0.0f;
                for(int k = 0; k < m_pq_m; k++)
                {
                    score += PQ_table[k][m_ivfList[vw][j].PQindex[k]];
                }
                matchScore.at(f).at(m_ivfList[vw][j].videoId) = min(score, matchScore.at(f).at(m_ivfList[vw][j].videoId));
            }
            delete[] feat_res;
            Delete2DArray(PQ_table);
            
        }
        
    }
    
    printf("average number of candidate frames: %f\n", aver_cand/(1.0*m_frameNum));
    
    Delete2DArray(m_ppFeat);
}

void IVFOPQ::reorder(float* feat)
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

void IVFOPQ::LoadSingleFeatFile(string srcFile, float**& m_ppFeat, int& m_frameNum)
{
    ifstream fin(srcFile, ios::binary);
    if(!fin.is_open())
    {
        cout << "Error open the feat file: " << srcFile << endl;
        return;
    }
    else
    {
        fin.seekg(0,ios::end);
        int file_size = (int)fin.tellg();
        fin.seekg(0,ios::beg);
        m_frameNum = file_size/(sizeof(float)*m_featDim);
        Init2DArray(m_ppFeat, m_frameNum, m_featDim);
        fin.read((char*)m_ppFeat[0], file_size);
        fin.close();
    }
    for(int i = 0; i < m_frameNum; i++){
        reorder(m_ppFeat[i]);
    }
}

void IVFOPQ::LoadIndex(string srcFile)
{
    cout<<"load index..."<<endl;
    ifstream fin(srcFile, ios::binary);
    if(!fin.is_open())
    {
        cout<<"Can not open the index file."<<endl;
        exit(0);
    }
    else
    {
        fin.read((char*)&m_featDim, sizeof(int));
        fin.read((char*)&m_coarseK, sizeof(int));
        fin.read((char*)&m_pq_m, sizeof(int));
        fin.read((char*)&m_pq_k, sizeof(int));
        m_pq_step = m_featDim/m_pq_m;
        fin.read((char*)&m_imgNum, sizeof(int));
        
        Init2DArray(m_ppCoarseCluster, m_coarseK, m_featDim);
        fin.read((char*)m_ppCoarseCluster[0], sizeof(float)*m_featDim*m_coarseK);
        
        m_prodQuantizer = new float**[m_pq_m];
        
        for(int i=0; i<m_pq_m; i++)
        {
            Init2DArray(m_prodQuantizer[i],m_pq_k,m_pq_step);
            fin.read((char*)m_prodQuantizer[i][0], sizeof(float)*m_pq_k*m_pq_step);
        }
        
        m_ivfSize = new int[m_coarseK];
        m_ivfList.reserve(m_coarseK);
        
        for(int i=0; i<m_coarseK; i++)
        {
            fin.read((char*)&m_ivfSize[i], sizeof(int));
            if(m_ivfSize[i]>0)
            {
                m_ivfList.at(i).reserve(m_ivfSize[i]);
                //m_ivfList.at(i).resize(m_ivfSize[i]);
                fin.read((char*)m_ivfList.at(i).data(), sizeof(IVFelem)*m_ivfSize[i]);
            }
        }
        
        m_imgLocation = new ImgNameStruct[m_imgNum];
        for(int i=0;i<m_imgNum;i++)
        {
            fin.read(m_imgLocation[i].ptr, sizeof(char)*max_path);
        }
        fin.close();
    }
}

void IVFOPQ::SaveIndex(string desDir)
{
    cout<<"save index file..."<<endl;
    string strImgNum, strK, strDim, str_pq_m, str_pq_k;
    stringstream sin_str;
    sin_str << m_imgNum;
    sin_str >> strImgNum;
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
    
    string m_desDir = desDir + "/OPQ_Index_db_" + strImgNum +"_dim_" + strDim + "_k_" + strK +"_PQ_m"+str_pq_m + "_k"+str_pq_k+ ".fvecs";
    ofstream outFile(m_desDir.c_str(), ios::binary);
    
    outFile.write((char*)&m_featDim, sizeof(int));
    outFile.write((char*)&m_coarseK, sizeof(int));
    outFile.write((char*)&m_pq_m, sizeof(int));
    outFile.write((char*)&m_pq_k, sizeof(int));
    outFile.write((char*)&m_imgNum, sizeof(int));
    outFile.write((char*)m_ppCoarseCluster[0], sizeof(float) * m_featDim * m_coarseK);
    
    for(int i=0;i<m_pq_m;i++)
    {
        outFile.write((char*)m_prodQuantizer[i][0], sizeof(float) * m_pq_step * m_pq_k);
    }
    
    ofstream fout("log.txt");
    for(int i = 0; i < m_coarseK; i++)
    {
        int n = (int)m_ivfList.at(i).size();
        
        if(n > 0) {
            fout << "coarse center: " << i << ", " << n << std::endl;
            std::cout << "coarse center: " << i << ", " << n << std::endl;
        }
        
        outFile.write((char*)&n, sizeof(int));
        for(int j = 0; j < n; j++)
        {
            outFile.write((char*)&m_ivfList.at(i).at(j).videoId, sizeof(int));
            for(int k = 0; k < m_pq_m; k++)
            {
                outFile.write((char*)&m_ivfList.at(i).at(j).PQindex[k], sizeof(uchar));
            }
        }
    }
    fout.close();
    for(int i = 0; i < m_imgNum; i++)
    {
        outFile.write(m_imgLocation[i].ptr, sizeof(char)*max_path);
    }
    
    outFile.close();
}
