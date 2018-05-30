
#include "train_PQ_codebook.h"

int main(int argc, char* argv[])
{
	if (argc != 9)
	{
		cout << "Error in input parameters!\n";
		return -1;
	}

	int maxSampeNum, k, featDim, pq_m, pq_k;

        string modelPath = string(argv[1]); // reorder模型文件路径
	string srcDir = string(argv[2]); // 特征文件路径
	string desDir = string(argv[3]); // 码本保存路径
	maxSampeNum = atoi(argv[4]); // 最大训练样本数目，可以大于实际的样本数目？
	k = atoi(argv[5]); // 一级码本数目
	featDim = atoi(argv[6]); // 特征维度
	pq_m = atoi(argv[7]); // 子段数目
	pq_k = atoi(argv[8]); // 二级子段码本数目

        std::cout << "hello world ... " << std::endl;
	TrainPQ m_train_pq(modelPath, maxSampeNum, featDim, k, pq_k, pq_m);

	m_train_pq.LoadFeatureSample(srcDir);
	m_train_pq.IFVPQ();
	m_train_pq.SaveCodebook(desDir);

	return 0;
}
