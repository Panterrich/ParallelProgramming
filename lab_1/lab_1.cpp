#include "user_mpi.h"
#include "worker.h"

int main(int argc, char** argv) 
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    // cpu_set_t mask;
    // CPU_ZERO(&mask);
    // CPU_SET(2 * rank, &mask);
    // sched_setaffinity(getpid(), sizeof(cpu_set_t), &mask);

    double start_time = MPI::Wtime();

    Worker worker(master.getRank(), master.getCommSize());
    
    worker.FillInitialConditions();
    worker.FillFirstLine(&master);
    worker.FillOtherLines(&master);

    if (master.getCommSize() > 1) 
    {
        worker.Gather(&master);
    }

    double end_time = MPI::Wtime();

    if (master.getRank() == 0) 
    {
        printf("Time: %.5lf\n", end_time - start_time);
#define ENABLE_SAVING_PICTURE
        #ifdef ENABLE_SAVING_PICTURE
        const char* name = "output.txt";
        FILE* file = fopen(name, "w");
        worker.Dump(file);
        fclose(file);
        #endif
    }

    return 0;
}
