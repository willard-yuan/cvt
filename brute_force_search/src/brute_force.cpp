#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "brutoforce.hpp"
#include "space_ip.hpp"

using namespace std;
using namespace hnswlib;

int main(int argc, const char * argv[]) {
    
    int ddim = 128;
    int topK = 100;
    std::string db_path = "db.bin";
    std::string querys_path = "querys.bin";
    std::string index_path = "index.bin";
    std::string gtfile_path = "gt.txt";
    
    ofstream gtFile;
    gtFile.open(gtfile_path);
    
    std::ifstream fp;
    int num_db = 0;
    fp.open(db_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_db, sizeof(int));
    
    // 暴力搜索
    InnerProductSpace ipspace(ddim);
    BruteforceSearch<float> *bruteAlg;
    bruteAlg = new BruteforceSearch<float>(&ipspace, num_db);
    
    // 读取database
    int count = 0;
    float *feat = new float[ddim];
    std::vector<std::string> dbIds;
    for (int i = 0; i < num_db; i++) {
        // 读取id
        int idSize = 0;
        fp.read((char*)&idSize, sizeof(int));
        char idName[1024] = {""};
        fp.read(idName, idSize);
        std::string idStr(idName);
        
        // 读取特征
        int dim_feat = 0;
        fp.read((char*)&dim_feat, sizeof(int));
        if (dim_feat != ddim) {
            cout << "file error";
            exit(1);
        }

        fp.read(reinterpret_cast<char *>(feat), dim_feat*sizeof(float));
        
        bruteAlg->addPoint((void *)feat, (size_t)count);
        dbIds.push_back(idStr);
        ++count;
        if((i+1)%100000 == 0) printf("indexed %d points\n", (i+1));
    }
    fp.close();
    bruteAlg->saveIndex(index_path);
    printf("brute force index finished building\n");
    
    // 读取querys
    int num_querys = 0;
    fp.open(querys_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_querys, sizeof(int));
    for (int i = 0; i < num_querys; i++) {
        // 读取id
        int idSize = 0;
        fp.read((char*)&idSize, sizeof(int));
        char idName[1024] = {""};
        fp.read(idName, idSize);
        std::string qIDStr(idName);
        
        // 读取特征
        int dim_feat = 0;
        fp.read((char*)&dim_feat, sizeof(int));
        if (dim_feat != ddim) {
            cout << "file error";
            exit(1);
        }
        fp.read(reinterpret_cast<char *>(feat), dim_feat*sizeof(float));
        
        std::priority_queue<std::pair<float, labeltype >> result = bruteAlg->searchKnn((void *)feat, topK);
        
        // 逆序处理
        std::vector<std::pair<std::string, float>> reverse_result;
        reverse_result.clear();
        while (!result.empty()) {
            reverse_result.push_back(std::pair<std::string, float>(dbIds[result.top().second].c_str(), 1.0-result.top().first));
            result.pop();
        }
        
        std::stringstream ids;
        std::stringstream dists;
        for(int i = (int)reverse_result.size()-1; i >= 0; i--){
            ids << reverse_result.at(i).first.c_str() << " ";
            dists << reverse_result.at(i).second << " ";
        }
        std::string idsStr = ids.str();
        std::string distsStr = dists.str();
        //printf("%s topK: %s dists: %s\n", qIDStr.c_str(), idsStr.c_str(), distsStr.c_str());
        
        gtFile << qIDStr.c_str() << " " << "topK: " << idsStr.c_str() << "dists: " << distsStr.c_str() << "\n";
        if((i+1)%1000 == 0) printf("searched %d points\n", (i+1));
    }
    gtFile.close();
    
    return 0;
}
