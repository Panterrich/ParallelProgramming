#include <omp.h>
#include <stdio.h>

int main()
{
    int res = 0;

    #pragma omp parallel for ordered
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        #pragma omp ordered
        {
            printf("%3d (%2d/%d)\n", res++,
                   omp_get_thread_num(), omp_get_num_threads());
        }
    }
}
