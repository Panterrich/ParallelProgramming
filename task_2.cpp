#include "include/UserMpi.h"

int main(int argc, char* argv[])
{
    if (argc != 2) 
    {
        printf("Enter the number N\n"
               "For example: ./a.out 10\n");
        return 0;
    }

    char* end = nullptr;
    unsigned long N = strtoul(argv[1], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    unsigned long sectionSize = N / master.getCommSize();
    unsigned long remains     = N % master.getCommSize();

    unsigned long beginPoint = 0;
    unsigned long   endPoint = 0;
    

    if (master.getRank() < static_cast<int>(remains))
    {
        beginPoint =  master.getRank()      * (sectionSize + 1);
          endPoint = (master.getRank() + 1) * (sectionSize + 1);
    }
    else
    {
        beginPoint =  master.getRank()      * sectionSize + remains;
          endPoint = (master.getRank() + 1) * sectionSize + remains;
    }

    double res = 0;

    for (unsigned long i = beginPoint; i < endPoint; i++)
    {
        res += 1.0 / (i + 1);
    }

    if (master.getRank() == 0)
    {
        double data = 0;

        for (int i = 0; i < master.getCommSize() - 1; i++)
        {
            master.recv(&data, 1, MPI_2REAL, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
            if (master.check()) return 1;

            res += data;
        }

        printf("The sum of the first %lu terms of the harmonic series is equal to %lg\n", N, res);
    }

    else
    {
        master.send(&res, 1, MPI_2REAL, 0, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    return 0;
}
