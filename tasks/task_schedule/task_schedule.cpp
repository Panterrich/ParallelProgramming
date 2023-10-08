#include <omp.h>
#include <stdio.h>

static constexpr unsigned int kNumThreads = 4;
static constexpr unsigned int kIterations = 65;

void foo(unsigned int value);

void static_schelude(unsigned int value);

void dynamic_schelude(unsigned int value);

void guided_schelude(unsigned int value);

void default_schelude();

int main()
{
    // статически распределяются задачи группами по value
    static_schelude(1);
    static_schelude(4);

    // динамически распределяются задачи группами по value
    dynamic_schelude(1);
    dynamic_schelude(4);

    // динамически распределяются задачи группами с динамическим шагом, уменьшающихся вплоть до value
    guided_schelude(1);
    guided_schelude(4);

    default_schelude();
}

//==============================================================================

void foo(unsigned int value)
{
    volatile unsigned int i = 0;

    for (; i < value; i++) {}

    printf("Thread #%-2d [%-2d]\n", omp_get_thread_num(), i);
}


void static_schelude(unsigned int value)
{
    printf("schelude(static, %d)\n", value);

    #pragma omp parallel for schedule(static, value) num_threads(kNumThreads)
    for (unsigned int i = 0; i < kIterations; i++)
    {
        foo(i);
    }

    printf("\n\n");
}

void dynamic_schelude(unsigned int value)
{
    printf("schelude(dynamic, %d)\n", value);

    #pragma omp parallel for schedule(dynamic, value) num_threads(kNumThreads)
    for (unsigned int i = 0; i < kIterations; i++)
    {
        foo(i);
    }

    printf("\n\n");
}

void guided_schelude(unsigned int value)
{
    printf("schelude(guided, %d)\n", value);

    #pragma omp parallel for schedule(guided, value) num_threads(kNumThreads)
    for (unsigned int i = 0; i < kIterations; i++)
    {
        foo(i);
    }

    printf("\n\n");
}

void default_schelude()
{
    printf("default\n");

    #pragma omp parallel for num_threads(kNumThreads)
    for (unsigned int i = 0; i < kIterations; i++)
    {
        foo(i);
    }

    printf("\n\n");
}
