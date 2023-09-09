#include <omp.h>
#include <stdio.h>
#include <err.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

bool ArgvToInt(unsigned long* value, const char* str)
{
    assert(value != nullptr);
    assert(str   != nullptr);

    char* end = nullptr;
    *value = strtoul(str, &end, 10);

    return (errno != ERANGE) && (*end == '\0');
}

int main(const int argc, const char* argv[])
{
    if (argc != 3)
    {
        printf("Enter the number N and num threads\n"
               "For example: ./a.out 10 5\n");
        return 0;
    }

    unsigned long N          = 0;
    unsigned long numThreads = 0;

    if (!ArgvToInt(&N,          argv[1])) return 1;
    if (!ArgvToInt(&numThreads, argv[2])) return 1;

    if (N < 2 * numThreads) numThreads = N / 2;

    omp_set_num_threads(numThreads);

    double sum = 0;

    #pragma omp parallel for reduction(+: sum)
        for (unsigned long i = 1; i <= N; i++)
        {
            sum += 1.0f / i;
        }

    printf("The sum of the first %lu terms of the harmonic series is equal to %lg\n", N, sum);
}
