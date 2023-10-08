#include <omp.h>
#include <stdio.h>

static constexpr unsigned int kNumThreads = 2;

void Dump(int level, int prevNumThreads, int globalNumThread)
{
    #pragma omp ordered
    {
        printf("Level %2d: prev num threads %d (%2d/%2d): %2d\n",
                level, prevNumThreads,
                omp_get_thread_num(), omp_get_num_threads(),
                globalNumThread);
    }
}

int main()
{
    omp_set_dynamic(false);
    omp_set_nested(true);

    int globalNumThread = 0;

    Dump(0, 0, globalNumThread + (1 << 0));

    int level0 = 1;

    #pragma omp parallel for ordered firstprivate(globalNumThread) num_threads(kNumThreads)
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        int level1 = omp_get_num_threads();

        globalNumThread += omp_get_thread_num();

        Dump(1, level0, globalNumThread + (1 << 1));

        globalNumThread <<= 1;

        #pragma omp parallel for ordered firstprivate(globalNumThread) num_threads(kNumThreads)
        for (int j = 0; j < omp_get_num_threads(); j++)
        {
            int level2 = omp_get_num_threads();

            globalNumThread += omp_get_thread_num();

            Dump(2, level1, globalNumThread + (1 << 2));

            globalNumThread <<= 1;

            #pragma omp parallel for ordered firstprivate(globalNumThread) num_threads(kNumThreads)
            for (int k = 0; k < omp_get_num_threads(); k++)
            {
                globalNumThread += omp_get_thread_num();

                Dump(3, level2, globalNumThread + (1 << 3));
            }
        }
    }
}
