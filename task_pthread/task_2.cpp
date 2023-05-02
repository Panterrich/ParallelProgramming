#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <stdlib.h>
#include <errno.h>

void* counter(void* arg);

struct ThreadData
{
    unsigned long start;
    unsigned long end;
};

int main(int argc, char* argv[])
{
    if (argc != 3) 
    {
        printf("Enter K number of threads and N number\n"
               "For example: ./a.out 10\n");
        return 0;
    }

    char* end = nullptr;
    unsigned long K = strtoul(argv[1], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    unsigned long N = strtoul(argv[2], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    std::vector<pthread_t>  pthreads(K);
    std::vector<ThreadData> data(K);

    int error = 0;

    unsigned long sectionSize = N / K;
    unsigned long remains     = N % K;

    for (unsigned long i = 0; i < K; i++)
    {
        if (i < remains)
        {
            data[i].start =  i      * (sectionSize + 1);
            data[i].end   = (i + 1) * (sectionSize + 1);
        }
        else
        {
            data[i].start =  i      * sectionSize + remains;
            data[i].end   = (i + 1) * sectionSize + remains;
        }

        error = pthread_create(&pthreads[i], NULL, counter, &data[i]);
        if (error)
        {
            perror("pthread_create");
            return 1;
        }
    }

    double* recv = 0;
    double sum  = 0;

    for (unsigned long i = 0; i < K; i++)
    {
        error = pthread_join(pthreads[i], (void**)&recv);
        if (error)
        {
            perror("pthread_join");
            return 1;
        }

        sum += *recv;

        delete recv;
    }

    printf("The sum of the first %lu terms of the harmonic series is equal to %lg\n", N, sum);
    
    return 0;
}

void* counter(void* arg)
{
    ThreadData* data = (ThreadData*) arg;

    double* res = new double(0);

    for (unsigned long i = data->start; i < data->end; i++)
    {
        *res += 1.0 / (i + 1);
    }

    pthread_exit(res);
}
