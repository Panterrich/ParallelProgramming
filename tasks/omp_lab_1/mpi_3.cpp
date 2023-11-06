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

    double* a            = nullptr;
    double* b            = nullptr;
    double* a_ref        = nullptr;
    double* b_ref        = nullptr;
    double* recv         = nullptr;
    int*    displs_1     = nullptr;
    int*    displs_2     = nullptr;
    int*    sendcounts_1 = nullptr;
    int*    sendcounts_2 = nullptr;

    size_t sectionSize = kISize / master.getCommSize();
    size_t remains     = kISize % master.getCommSize();
    size_t size        = sectionSize + (master.getRank() < static_cast<int>(remains));

    size *= kJSize;

    if (master.getRank() == 0)
    {
        a     = new double[kISize * kJSize];
        b     = new double[kISize * kJSize];
        a_ref = new double[kISize * kJSize];
        b_ref = new double[kISize * kJSize];

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
            {
                a_ref[i * kJSize + j] = 10 * i + j;
                b_ref[i * kJSize + j] = 0;
            }

        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
                a_ref[i * kJSize + j] = std::sin(0.1 * a_ref[i * kJSize + j]);

        for (size_t i = 0; i < kISize - 1; i++)
            for (size_t j = 0; j < kJSize; j++)
                b_ref[i * kJSize + j] = a_ref[(i + 1) * kJSize + j] * 1.5;


        for (size_t i = 0; i < kISize; i++)
            for (size_t j = 0; j < kJSize; j++)
            {
                a[i * kJSize + j] = 10 * i + j;
                b[i * kJSize + j] = 0;
            }

        displs_1     = new int[master.getCommSize()];
        displs_2     = new int[master.getCommSize()];
        sendcounts_1 = new int[master.getCommSize()];
        sendcounts_2 = new int[master.getCommSize()];

        for (int i = 0; i < master.getCommSize(); i++)
        {
            displs_1[i]     = (i < static_cast<int>(remains)) ? (sectionSize + 1) * i : sectionSize * i + remains;
            sendcounts_1[i] = sectionSize + (i < static_cast<int>(remains));

            displs_1[i]     *= kJSize;
            sendcounts_1[i] *= kJSize;
        }

        for (int i = 0; i < master.getCommSize(); i++)
        {
            displs_2[i]     = (i < static_cast<int>(remains)) ? (sectionSize + 1) * i : sectionSize * i + remains;
            sendcounts_2[i] = sectionSize + (i < static_cast<int>(remains));

            displs_2[i]     *= kJSize;
            sendcounts_2[i] *= kJSize;
        }

        sendcounts_2[master.getCommSize() - 1] -= kJSize;
    }

    recv = new double[size];

    master.barrier(MPI_COMM_WORLD);

    double time  = 0;
    double start = MPI_Wtime();

    master.scatterv(a, sendcounts_1, displs_1, MPI_DOUBLE, recv, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;


    time -= MPI_Wtime();

    for (size_t i = 0; i < size; i++)
        recv[i] = std::sin(0.1 * recv[i]);

    master.gatherv(recv, size, MPI_DOUBLE, a, sendcounts_1, displs_1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;

    time += MPI_Wtime();

    if (master.getRank() == master.getCommSize() - 1)
        size -= kJSize;

    master.scatterv(a + kJSize, sendcounts_2, displs_2, MPI_DOUBLE, recv, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;

    time -= MPI_Wtime();

    for (size_t i = 0; i < size; i++)
        recv[i] = recv[i] * 1.5;

    master.gatherv(recv, size, MPI_DOUBLE, b, sendcounts_2, displs_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (master.check()) return 1;

    double end = MPI_Wtime();
    time += end;

    if (master.getRank() == 0)
    {
        assert(memcmp(a, a_ref, sizeof(a[0]) * kISize * kJSize) == 0);
        assert(memcmp(b, b_ref, sizeof(b[0]) * kISize * kJSize) == 0);

        printf("Time:                   %.8lf\n", end - start);
        printf("Time (without scatter): %.8lf\n", time);
    }

    delete [] a;
    delete [] b;
    delete [] recv;
    delete [] displs_1;
    delete [] displs_2;
    delete [] sendcounts_1;
    delete [] sendcounts_2;
}
