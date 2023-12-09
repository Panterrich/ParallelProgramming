#include <assert.h>
#include <vector>
#include <cstring>
#include <type_traits>
#include <random>

#include "user_mpi.h"

static const size_t kISize = 20000;
static const size_t kJSize = 20000;

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

    size_t sectionSize = kISize / master.getCommSize();
    size_t remains     = kISize % master.getCommSize();
    size_t size        = sectionSize + (master.getRank() < static_cast<int>(remains));

    size *= kJSize;

    if (master.getRank() == 0)
    {
        a = new double[kISize * kJSize];
        b = new double[kISize * kJSize];

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                b[i * kJSize + j] = 10 * i + j;

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                b[i * kJSize + j] = std::sin(2 * b[i * kJSize + j]);

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a[i * kJSize + j] = 10 * i + j;

        displs     = new int[master.getCommSize()];
        sendcounts = new int[master.getCommSize()];

        for (int i = 0; i < master.getCommSize(); i++)
        {
            displs[i]     = (i < static_cast<int>(remains)) ? (sectionSize + 1) * i : sectionSize * i + remains;
            sendcounts[i] = sectionSize + (i < static_cast<int>(remains));

            displs[i]     *= kJSize;
            sendcounts[i] *= kJSize;
        }
    }

    recv = new double[size];

    master.barrier(MPI_COMM_WORLD);

    double time  = 0;
    double start = MPI_Wtime();

    master.scatterv(a, sendcounts, displs, MPI_DOUBLE, recv, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;

    time -= MPI_Wtime();

    for (size_t i = 0; i < size; i++)
        recv[i] = std::sin(2 * recv[i]);

    master.gatherv(recv, size, MPI_DOUBLE, a, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    time += end;

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
