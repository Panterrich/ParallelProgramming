#include "user_mpi.h"

int main(int argc, char* argv[])
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    int res = 0;

    if (master.getCommSize() == 1) 
    {
        printf("My rank = %d: i = %d\n", master.getRank(), res);
        return 0;
    }

    if (master.getRank() == 0)
    {
        printf("My rank = %d: i = %d\n", master.getRank(), res);

        master.send(&res, 1, MPI_INT, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        master.recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        res++;

        printf("My rank = %d: i = %d\n", master.getRank(), res);
    }

    else if (master.getRank() == master.getCommSize() - 1)
    {
        master.recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        res++;

        printf("My rank = %d: i = %d\n", master.getRank(), res); 

        master.send(&res, 1, MPI_INT, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    else
    {
        master.recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        res++;

        printf("My rank = %d: i = %d\n", master.getRank(), res); 

        master.send(&res, 1, MPI_INT, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        master.recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        res++;

        printf("My rank = %d: i = %d\n", master.getRank(), res);

        master.send(&res, 1, MPI_INT, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    return 0;
}
