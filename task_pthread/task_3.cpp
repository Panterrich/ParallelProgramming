#include "user_mpi.h"

#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <stdlib.h>
#include <errno.h>

void* counter(void* arg);

struct ThreadData
{
    int tid;
};

pthread_mutex_t mutex;
pthread_cond_t cond;
int current_tid = 0;

int x = 0;

int main(int argc, char* argv[])
{
    if (argc != 2) 
    {
        printf("Enter K number of threads\n"
               "For example: ./a.out 10\n");
        return 0;
    }

    char* end = nullptr;
    unsigned long K = strtoul(argv[1], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    std::vector<pthread_t>  pthreads(K);
    std::vector<ThreadData> data(K);

    pthread_mutex_init(&mutex, NULL);

    int error = 0;

    for (unsigned long i = 0; i < K; i++)
    {
        data[i].tid = i;

        error = pthread_create(&pthreads[i], NULL, counter, &data[i]);
        if (error)
        {
            perror("pthread_create");
            return 1;
        }
    }

    for (unsigned long i = 0; i < K; i++)
    {
        error = pthread_join(pthreads[i], NULL);
        if (error)
        {
            perror("pthread_join");
            return 1;
        }
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}

void* counter(void* arg)
{
    ThreadData* data = (ThreadData*) arg;

    pthread_mutex_lock(&mutex);
    
    while (data->tid != current_tid)
    {
        pthread_cond_wait(&cond, &mutex);
    }

    printf("Rank %d value %d\n", data->tid, x++);

    current_tid++;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}
