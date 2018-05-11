
#include "common.h"
#include "IVFOPQ.h"

const string modelFilePath = "/Users/willard/projects/IVFOPQ/model/OPQ_db_5950000_dim_128_k_8192_PQ_m16_k256_reorder.model";

#if 1
int main(int argc, char* argv[])
{
    string modelFile = modelFilePath; // OPQ训练模型
    string imgLists = "/Users/willard/codes/cpp/multi_frame_search/opq_test_online/5_feats_list.txt"; // 特征存放的路径，txt文件，每一行对应特征的路径
    string desDir = "/Users/willard/codes/cpp/multi_frame_search/opq_test_online/"; // 索引保存的路径
    int maxImageNum = 1000; //atoi(100); // 索引对应的最大的图片数量
    
    // 读取视频特征文件路径
    vector<string> featFiles;
    get_vector_of_strings_from_file_lines(imgLists, featFiles);
    int num = (int)featFiles.size();
    cout << num << endl;
    
    // 构建索引
    IVFOPQ index(maxImageNum);
    index.LoadModel(modelFile);
    index.IndexDatabase(featFiles);
    index.SaveIndex(desDir);
    
    return 0;
}
#endif

#if 0
int main(int argc, char* argv[])
{
    int num_nearest = 3; // 查找prob的数目
    int num_show = 5;  // top@K
    
    string srcFile = "/Users/willard/codes/cpp/multi_frame_search/opq_test_online/OPQ_Index_db_5_dim_128_k_8192_PQ_m16_k256.fvecs"; // 索引文件路径
    string queryImgPath = "/Users/willard/codes/cpp/multi_frame_search/opq_test_online/query/6231519245_6_feat.bin"; // 查询
    string desFile = "/Users/willard/codes/cpp/multi_frame_search/opq_test_online/result.txt"; // 保存搜索结果
    
    vector<string> queryPaths;
    queryPaths.push_back(queryImgPath);
    int n = (int)queryPaths.size();
    
    IVFOPQ ivfpq_search;
    ivfpq_search.LoadModel(modelFilePath);
    ivfpq_search.LoadIndex(srcFile);

    ofstream fout(desFile.c_str());
    
    for(int i = 0; i < n; i++)
    {
        vector<vector<float> > score;
        ivfpq_search.Query(queryPaths.at(i), score, num_nearest);
        
        int frame_num = (int)score.size();
        if(frame_num > 0)
        {
            int img_num = (int)score.at(0).size();
            vector<float> score_total(img_num, 0.0f);
            for(int j = 0; j < frame_num; j++)
            {
                for(int k = 0; k < img_num; k++)
                {
                    score_total.at(k) += score.at(j).at(k);
                }
            }
            vector<pair<float, uint> > result = get_sort_results(score_total, num_show);
            
            fout << queryPaths.at(i) << "  " << i << endl;
            for(int j = 0; j < num_show; j++)
            {
                fout << get_base_name(ivfpq_search.m_imgLocation[result.at(j).second].ptr) << " ";
            }
            fout << endl;
            
            for(int j = 0; j < num_show; j++)
            {
                fout << result.at(j).first << " ";
                std::cout << result.at(j).second << ": " << result.at(j).first << std::endl;
            }
            fout << "\n\n";
        }
        
        
        cout << "query ID: " << i <<", frame_num:"<< frame_num <<endl;
        
    }
    fout.close();
    return 0;
}
#endif
