#include <assert.h>
#include <vector>
#include <cstring>
#include <type_traits>
#include <random>

#include "user_mpi.h"

static const size_t kISize = 10000;
static const size_t kJSize = 10000;

/**
 *  D (1, -1) => d (<, >) => потоковая зависимость
 *
 */

int main(int argc, char* argv[])
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    double* a          = nullptr;
    double* b          = nullptr;
    double* recv       = nullptr;
    int*    displs     = nullptr;
    int*    sendcounts = nullptr;

    size_t sectionSize = (kJSize - 1) / master.getCommSize();
    size_t remains     = (kJSize - 1) % master.getCommSize();
    size_t size        = sectionSize + (master.getRank() < static_cast<int>(remains));

    if (master.getRank() == 0)
    {
        a = new double[kISize * kJSize];
        b = new double[kISize * kJSize];

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                b[i * kJSize + j] = 10 * i + j;

        for (size_t i = 1; i < kISize; i++)
            for (size_t j = 0; j < kJSize - 1; j++)
                b[i * kJSize + j] = std::sin(2 * b[(i - 1) * kJSize + (j + 1)]);

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a[i * kJSize + j] = 10 * i + j;

        displs     = new int[master.getCommSize()];
        sendcounts = new int[master.getCommSize()];

        for (int i = 0; i < master.getCommSize(); i++)
        {
            displs[i]     = (i < static_cast<int>(remains)) ? (sectionSize + 1) * i : sectionSize * i + remains;
            sendcounts[i] = sectionSize + (i < static_cast<int>(remains));
        }
    }

    recv = new double[size];

    master.barrier(MPI_COMM_WORLD);

    double time  = 0;
    double start = MPI_Wtime();

    for (size_t i = 1; i < kISize; i++)
    {
        master.scatterv(a + (i - 1) * kJSize + 1, sendcounts, displs, MPI_DOUBLE, recv, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        time -= MPI_Wtime();

        for (size_t j = 0; j < size; j++)
            recv[j] = std::sin(2 * recv[j]);

        master.gatherv(recv, size, MPI_DOUBLE, a + i * kJSize, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        time += MPI_Wtime();

        master.barrier(MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();

    if (master.getRank() == 0)
    {
        assert(memcmp(a, b, sizeof(a[0]) * kISize * kJSize) == 0);

        printf("Time:                   %.8lf\n", end - start);
        printf("Time (without scatter): %.8lf\n", time);
    }

    delete [] a;
    delete [] b;
    delete [] recv;
    delete [] displs;
    delete [] sendcounts;
}
