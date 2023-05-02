#include <stdio.h>
#include <pthread.h>
#include <vector>
#include <stdlib.h>
#include <errno.h>

void* hello_world(void* arg);

struct ThreadData
{
    int tid;
};

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

    int error = 0;

    for (unsigned long i = 0; i < K; i++)
    {
        data[i].tid = i;

        error = pthread_create(&pthreads[i], NULL, hello_world, &data[i]);
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
    
    return 0;
}

void* hello_world(void* arg)
{
    printf("Hello World! My number %d\n", *(int*)arg);

    pthread_exit(nullptr);
}

