
#ifndef common_h
#define common_h

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

void static get_vector_of_strings_from_file_lines(const std::string file_name,
                                           std::vector<std::string>& out) {
    std::ifstream in_file(file_name.c_str());
    std::string line;
    out.clear();
    while (!in_file.eof()) {
        if (getline(in_file, line)) out.push_back(line);
    }
    in_file.close();
}



std::vector<std::pair<float, uint>> static get_sort_results(const std::vector<float>& match_score, int results_per_query)
{
    int num = (int)match_score.size();
    std::vector<std::pair<float, uint> > results(num);
    std::vector<std::pair<float, uint> > t(results_per_query);
    for (uint i = 0; i < num; i++)
    {
        results.at(i) = std::make_pair(match_score.at(i), i);
    }
    //partial_sort(results.begin(), results.begin() + 200, results.end(), cmp_float_uint_descend);
    std::partial_sort_copy(results.begin(), results.end(), t.begin(), t.end());
    return t;
}

std::string static get_base_name(const std::string path)
{
    size_t pos = path.find_last_of("/\\");
    std::string str1 = path.substr(pos+1);
    pos = str1.find_last_of('.');
    std::string str2 = str1.substr(0,pos);
    return str2;
}

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

template<typename T> void print2DArray(T** &f, int row, int col){
    for(int i = 0; i < row; i ++){
        if(f[i] != NULL){
            for(int j = 0; j < col; j++){
                std::cout << f[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

#endif /* common_h */
