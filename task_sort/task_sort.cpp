#include <vector>
#include <type_traits>

#include "user_mpi.h"

static constexpr size_t N = 1000;

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

    int mid = (left + right) / 2;

    mergeSort(array, left, mid, buffer);
    mergeSort(array, mid + 1, right, buffer);

    merge(array, left, mid, right, buffer);
}

template <typename T>
void mergeSort(T* array, int left, int right)
{
    std::vector<T> buffer(right - left + 1);

    mergeSort(array, left, right, buffer);
}


int main(int argc, char* argv[])
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    char* end_str = nullptr;
    size_t ARRAY_SIZE = strtoul(argv[1], &end_str, 10);

    if ((errno == ERANGE) || (*end_str != '\0'))
        return 1;

    unsigned long sectionSize = ARRAY_SIZE / master.getCommSize();
    unsigned long remains     = ARRAY_SIZE % master.getCommSize();
    unsigned long size        = sectionSize + (master.getRank() < static_cast<int>(remains));

    int* array      = nullptr;
    int* displs     = nullptr;
    int* sendcounts = nullptr;

    double start = 0.f;
    double end   = 0.f;

    if (master.getRank() == 0)
    {
        array = new int[ARRAY_SIZE]{};
        if (!array) return -1;
        
        for (size_t i = 0; i < ARRAY_SIZE; i++)
        {
            array[i] = i % N;
        }

        displs     = new int[master.getCommSize()];
        sendcounts = new int[master.getCommSize()];

        for (int i = 0; i < master.getCommSize(); i++)
        {
            displs[i]     = (i < static_cast<int>(remains)) ? (sectionSize + 1) * i : sectionSize * i + remains;
            sendcounts[i] = sectionSize + (i < static_cast<int>(remains));
        }
    }

    int* sub_array = new int[size]{};
    std::vector<int> buffer(ARRAY_SIZE);

    start = MPI_Wtime();

    master.scatterv(array, sendcounts, displs, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;

    mergeSort(sub_array, 0, size - 1);

    int processes = master.getCommSize();
    int m = 1;

    unsigned long size_recv = 0;

    while (processes > 1)
    {
        processes = processes / 2 + (processes & 1);

        if ((master.getRank() - m) % (2 * m) == 0)
        {
            master.send(&size, 1, MPI_UNSIGNED_LONG, master.getRank() - m, 0, MPI_COMM_WORLD);
            if (master.check()) return 1;

            master.send(sub_array, size, MPI_INT, master.getRank() - m, 0, MPI_COMM_WORLD);
            if(master.check()) return 1;
        }

        if ((master.getRank() % (2 * m) == 0) && (master.getCommSize() - master.getRank() > m))
        {
            master.recv(&size_recv, 1, MPI_UNSIGNED_LONG, master.getRank() + m, MPI_ANY_TAG, MPI_COMM_WORLD);
            if (master.check()) return 1;

            int* sorted_array = new int[size + size_recv];

            master.recv(sorted_array, size_recv, MPI_INT, master.getRank() + m, MPI_ANY_TAG, MPI_COMM_WORLD);
            if (master.check()) return 1;

            for (unsigned long i = 0; i < size; i++)
                sorted_array[size_recv + i] = sub_array[i];

            merge(sorted_array, 0, size_recv - 1, size + size_recv - 1, buffer);
            
            delete[] sub_array;

            sub_array = sorted_array;
            size = size + size_recv;
        }

        m <<= 1;
    }

    end = MPI_Wtime();

    if (master.getRank() == 0)
    {
        printf("Time: %.8lf\n", end - start);

        for (size_t i = 0; i < ARRAY_SIZE; i++)
        {
            printf("[%lu] %d\n", i, sub_array[i]);
        }
    }

    delete[] array;
    delete[] sub_array;
    delete[] displs;
    delete[] sendcounts;
}
