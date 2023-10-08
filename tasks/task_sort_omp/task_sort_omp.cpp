#include <omp.h>
#include <assert.h>
#include <vector>
#include <type_traits>
#include <random>
#include <benchmark/benchmark.h>

static constexpr size_t kMaxThreadsNum = 6;
static constexpr size_t kMinSize =  1ULL << 20;
static constexpr size_t kMaxSize = 10ULL << 20;
static constexpr size_t kStepSize = 2ULL << 20;

int TASK_SIZE = 1 << 11;
int INSERTION_SIZE = 64;

template <typename T>
void insertionSort(T* arr, int left, int right)
{
    T key = {};

    int j = left;

    for (int i = left + 1; i <= right; i++)
    {
        key = arr[i];
        j = i - 1;

        // Move elements of arr[left...i-1], that are greater than key,
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }

        arr[j + 1] = key;
    }
}

template <typename T>
void merge(T* array, int left, int mid, int right, std::vector<T>& buffer) {
    int left_size  = mid - left + 1;
    int right_size = right - mid;

    for (int i = 0; i < left_size; i++)
    {
        buffer[i] = array[left + i];
    }

    for (int j = 0; j < right_size; j++)
    {
        buffer[left_size + j] = array[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;

    while (i < left_size && j < right_size)
    {
        if (buffer[i] <= buffer[left_size + j])
        {
            array[k++] = buffer[i++];
        }
        else
        {
            array[k++] = buffer[left_size + j++];
        }
    }

    while (i < left_size)
    {
        array[k++] = buffer[i++];
    }

    while (j < right_size)
    {
        array[k++] = buffer[left_size + j++];
    }
}

template <typename T>
void mergeSort(T* array, int left, int right, std::vector<T>& buffer)
{
    if (left >= right) return;

    if (right - left > INSERTION_SIZE)
    {
        int mid = (left + right) / 2;

        #pragma omp taskgroup
        {
            #pragma omp task shared(array, buffer) untied if (right - left >= TASK_SIZE)
            mergeSort(array, left, mid, buffer);

            #pragma omp task shared(array, buffer) untied if (right - left >= TASK_SIZE)
            mergeSort(array, mid + 1, right, buffer);

            #pragma omp taskyield
        }

        merge(array, left, mid, right, buffer);
    }
    else
    {
        insertionSort(array, left, right);

        // int mid = (left + right) / 2;

        // mergeSort(array, left, mid, buffer);
        // mergeSort(array, mid + 1, right, buffer);
        // merge(array, left, mid, right, buffer);
    }
}

template <typename T>
void mergeSort(T* array, int left, int right)
{
    std::vector<T> buffer(right - left + 1);

    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(array, left, right, buffer);
    }
}

static void Sort(benchmark::State& state)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    omp_set_num_threads(state.range(0));
    size_t size = state.range(1);
    // TASK_SIZE = state.range(2);
    INSERTION_SIZE = state.range(2);

    std::vector<int> vec(size);

    for (size_t i = 0; i < size; i++)
    {
        vec[i] = dist(rng);
    }

    // std::vector<int> vec1 = vec;
    // std::vector<int> vec2 = vec;

    // std::sort(vec1.begin(), vec1.end());
    // mergeSort(vec2.data(), 0, vec2.size() - 1);

    // assert(vec1 == vec2);

    for (auto _ : state)
    {
        //std::sort(vec.begin(), vec.end());
        mergeSort(vec.data(), 0, vec.size() - 1);
    }
}

// BENCHMARK(Sort)
//     ->ArgsProduct({
//       benchmark::CreateDenseRange(1, kMaxThreadsNum, /*step=*/1),
//       benchmark::CreateDenseRange(kMinSize, kMaxSize, /*step=*/kStepSize),
//     })
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();
//     // ->Repetitions(5);


// BENCHMARK(Sort)
//     ->ArgsProduct({
//       benchmark::CreateDenseRange(kMaxThreadsNum, kMaxThreadsNum, /*step=*/1),
//       benchmark::CreateDenseRange(kMaxSize, kMaxSize, /*step=*/kStepSize),
//       benchmark::CreateRange(1<<6, 1<<18, 2)
//     })
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime()
//     ->Repetitions(5);

BENCHMARK(Sort)
    ->ArgsProduct({
      benchmark::CreateDenseRange(kMaxThreadsNum, kMaxThreadsNum, /*step=*/1),
      benchmark::CreateDenseRange(kMaxSize, kMaxSize, /*step=*/kStepSize),
      benchmark::CreateRange(1<<1, 1<<15, 2)
    })
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Repetitions(5);

BENCHMARK_MAIN();
