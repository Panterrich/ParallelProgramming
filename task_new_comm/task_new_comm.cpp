#include "user_mpi.h"

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

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    MPI_Comm new_comm = {};
    int color = !!master.getRank();
    MPI_Comm_split(MPI_COMM_WORLD, color, master.getRank(), &new_comm);

    master.setCommSize(new_comm);
    if (master.check()) return 1;

    master.setRank(new_comm);
    if (master.check()) return 1;

    printf("COMM[%d]: My rank = %d \n", color, master.getRank());

    if (color == 0) return 0;

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

    double all = 0;

    master.reduce(&res, &all, 1, MPI_DOUBLE, MPI_SUM, 0, new_comm);
    if (master.check()) return 1;

    if (master.getRank() == 0)
    {
        printf("The sum of the first %lu terms of the harmonic series is equal to %lg\n", N, all);
    }

    return 0;
}
