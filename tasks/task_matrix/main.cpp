#include <omp.h>
#include <assert.h>
#include <chrono>
#include <random>

#include <benchmark/benchmark.h>

static constexpr size_t kMaxThreadsNum = 6;

#include "matrix.hpp"

// static void MatrixAddition(benchmark::State& state)
// {
//     std::mt19937 rng;
//     rng.seed(std::random_device()());
//     std::uniform_real_distribution<float> dist(-10, 10);

//     size_t size = state.range(1);

//     Matrix::Matrix<float> a(size, size);
//     Matrix::Matrix<float> b(size, size);
//     Matrix::Matrix<float> c{size, size};

//     for (size_t i = 0; i < size; i++)
//     {
//         for (size_t j = 0; j < size; j++)
//         {
//             a[i][j] = dist(rng);
//             b[i][j] = dist(rng);
//         }
//     }

//     for (auto _ : state)
//     {
//         omp_set_num_threads(state.range(0));
//         c = a + b;
//     }
// }


// BENCHMARK(MatrixAddition)
//     ->ArgsProduct({
//       benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
//       {384, 764, 1152}
//     })
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();


#ifdef TRANSFORM
static void MatrixMultiplicationCacheFriendly(benchmark::State& state)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    size_t size = state.range(1);

    Matrix::Matrix<float> a(size, size);
    Matrix::Matrix<float> b(size, size);
    Matrix::Matrix<float> c{size, size};

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            a[i][j] = dist(rng);
            b[i][j] = dist(rng);
        }
    }

    for (auto _ : state)
    {
        omp_set_num_threads(state.range(0));
        c = a * b;
    }
}

BENCHMARK(MatrixMultiplicationCacheFriendly)
    ->ArgsProduct({
      benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
      {384, 764, 1152}
    })
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Repetitions(10);

#else // TRANSFORM

static void MatrixMultiplication(benchmark::State& state)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    size_t size = state.range(1);

    Matrix::Matrix<float> a(size, size);
    Matrix::Matrix<float> b(size, size);
    Matrix::Matrix<float> c{size, size};

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            a[i][j] = dist(rng);
            b[i][j] = dist(rng);
        }
    }

    for (auto _ : state)
    {
        omp_set_num_threads(state.range(0));
        c = a * b;
    }
}

BENCHMARK(MatrixMultiplication)
    ->ArgsProduct({
      benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
      {384, 764, 1152}
    })
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Repetitions(10);

#endif // TRANSFORM

BENCHMARK_MAIN();
