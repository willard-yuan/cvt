#include <random>
#include <stdlib.h>

#include "../include/utils.h"


/*static std::mt19937 static_gen;
int pupiltracker::random(int min, int max)
{
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(static_gen);
}*/

/*int pupiltracker::random(int min, int max)
{
    init();
    //srand((unsigned)time(NULL));
    //static int seed = 0;
    //srand(seed++);

    double u = (((double) rand() / RAND_MAX) + 1)/2;
    return max+(max-min)*u;
}*/

/*int pupiltracker::random(int min, int max, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(gen);
}*/

int pupiltracker::random(int min, int max){
	return (int) rand() / (RAND_MAX + 1.0) * (max-min+1) + min;
}

int pupiltracker::random(int min, int max, unsigned int seed)
{
    srand(seed);
	return (int) rand() / (RAND_MAX + 1.0) * (max-min+1) + min;
}
