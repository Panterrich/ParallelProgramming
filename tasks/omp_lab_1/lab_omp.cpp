#include <omp.h>
#include <assert.h>
#include <vector>
#include <cstring>
#include <type_traits>
#include <random>
#include <benchmark/benchmark.h>

static const size_t kMaxThreadsNum = 6;

static const size_t kISize = 10000;
static const size_t kJSize = 10000;

static void TaskOmp0(benchmark::State& state)
{
    double* a = new double[kISize * kJSize];
    double* b = new double[kISize * kJSize];

    for (size_t i = 0; i < kISize; i++)
        for (size_t j = 0; j < kJSize; j++)
            b[i * kJSize + j] = 10 * i + j;

    for (size_t i = 0; i < kISize; i++)
        for (size_t j = 0; j < kJSize; j++)
            b[i * kJSize + j] = std::sin(2 * b[i * kJSize + j]);

    omp_set_num_threads(state.range(0));

    for (auto _ : state)
    {
        state.PauseTiming();
        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a[i * kJSize + j] = 10 * i + j;
        state.ResumeTiming();

        #pragma omp parallel for
        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a[i * kJSize + j] = std::sin(2 * a[i * kJSize + j]);

        state.PauseTiming();
        assert(memcmp(a, b, sizeof(a[0]) * kISize * kJSize) == 0);
        state.ResumeTiming();
    }

    delete [] a;
    delete [] b;
}

static void TaskOmp2(benchmark::State& state)
{
    double* a = new double[kISize * kJSize];
    double* b = new double[kISize * kJSize];

    for (size_t i = 0; i < kISize; i++)
        for (size_t j = 0; j < kJSize; j++)
            b[i * kJSize + j] = 10 * i + j;

    /** a[i][j] <- a[i + 3][j - 4] => D = (-3, 4)
     *                                d = ( >, <)
     *  По внешнему циклу антизависимость, нужно сохранять значения.
     *  По внутреннему потоковая зависимость.
     */

    for (size_t i = 0; i < kISize - 3; i++)
        for (size_t j = 4; j < kJSize; j++)
            b[i * kJSize + j] = std::sin(0.04 * b[(i + 3) * kJSize + (j - 4)]);

    omp_set_num_threads(state.range(0));

    for (auto _ : state)
    {
        state.PauseTiming();
        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a[i * kJSize + j] = 10 * i + j;
        state.ResumeTiming();

        #pragma omp parallel
        {
            size_t sectionSize = (kISize - 3) / omp_get_num_threads();
            size_t remains     = (kISize - 3) % omp_get_num_threads();

            size_t beginPoint = 0;
            size_t   endPoint = 0;


            if (omp_get_thread_num() < static_cast<int>(remains))
            {
                beginPoint =  omp_get_thread_num()      * (sectionSize + 1);
                endPoint   = (omp_get_thread_num() + 1) * (sectionSize + 1);
            }
            else
            {
                beginPoint =  omp_get_thread_num()      * sectionSize + remains;
                endPoint   = (omp_get_thread_num() + 1) * sectionSize + remains;
            }

            static const size_t kShift = 3;

            double* copy = nullptr;

            if (omp_get_thread_num() != omp_get_num_threads() - 1)
            {
                copy = new double[kJSize * kShift];
                memcpy(copy, a + endPoint * kJSize, kJSize * kShift * sizeof(a[0]));
            }

            #pragma omp barrier

            if (omp_get_thread_num() != omp_get_num_threads() - 1)
            {
                for (size_t i = beginPoint; i < endPoint - kShift; i++)
                {
                    for (size_t j = 4; j < kJSize; j++)
                        a[i * kJSize + j] = std::sin(0.04 * a[(i + 3) * kJSize + (j - 4)]);
                }

                size_t i0 = endPoint - kShift;

                for (size_t i = endPoint - kShift; i < endPoint; i++)
                {
                    for (size_t j = 4; j < kJSize; j++)
                        a[i * kJSize + j] = std::sin(0.04 * copy[(i - i0) * kJSize + (j - 4)]);
                }
            }
            else
            {
                for (size_t i = beginPoint; i < endPoint; i++)
                {
                    for (size_t j = 4; j < kJSize; j++)
                        a[i * kJSize + j] = std::sin(0.04 * a[(i + 3) * kJSize + (j - 4)]);
                }
            }

            delete [] copy;
        }

        state.PauseTiming();
        assert(memcmp(a, b, sizeof(a[0]) * kISize * kJSize) == 0);
        state.ResumeTiming();
    }

    delete [] a;
    delete [] b;
}

BENCHMARK(TaskOmp2)
    ->ArgsProduct({
      benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
    })
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
    // ->Repetitions(5);

BENCHMARK(TaskOmp0)
    ->ArgsProduct({
      benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
    })
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
    // ->Repetitions(5);



BENCHMARK_MAIN();

