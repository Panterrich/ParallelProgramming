#include "include/UserMpi.h"

int main(int argc, char* argv[])
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    printf("Hello World!\n");
    printf("Communicator size = %d My rank = %d\n", master.getCommSize(), master.getRank());

    return 0;
}
