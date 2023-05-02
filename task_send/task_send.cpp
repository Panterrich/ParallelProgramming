#include <unistd.h>
#include "user_mpi.h"

// A quick overview of MPI's send modes
// MPI has a number of different "send modes." These represent different choices of buffering (where is the data kept until it is received) and synchronization (when does a send complete). In the following, I use "send buffer" for the user-provided buffer to send.
// MPI_Send - performs a blocking send
//  MPI_Send will not return until you can use the send buffer. It may or may not block (it is allowed to buffer, either on the sender or receiver side, or to wait for the matching receive).
// MPI_Bsend - basic send with user-provided buffering
//  May buffer; returns immediately and you can use the send buffer. A late add-on to the MPI specification. Should be used only when absolutely necessary.
// MPI_Ssend - blocking synchronous send
//  will not return until matching receive posted
// MPI_Rsend - blocking ready send
//  May be used ONLY if matching receive already posted. User responsible for writing a correct program.
// MPI_Isend
//  Nonblocking send. But not necessarily asynchronous. You can NOT reuse the send buffer until either a successful, wait/test or you KNOW that the message has been received (see MPI_Request_free). Note also that while the I refers to immediate, there is no performance requirement on MPI_Isend. An immediate send must return to the user without requiring a matching receive at the destination. An implementation is free to send the data to the destination before returning, as long as the send call does not block waiting for a matching receive. Different strategies of when to send the data offer different performance advantages and disadvantages that will depend on the application.
// MPI_Ibsend
//  buffered nonblocking
// MPI_Issend
//  Synchronous nonblocking. Note that a Wait/Test will complete only when the matching receive is posted.
// MPI_Irsend
//  As with MPI_Rsend, but nonblocking.
// Note that "nonblocking" refers ONLY to whether the data buffer is available for reuse after the call. No part of the MPI specification, for example, mandates concurent operation of data transfers and computation.

static constexpr size_t min_size = 1;
static constexpr size_t max_size = 2; // 65480 ref for send and rsend
static constexpr size_t step     = 1000;
static constexpr size_t iters    = 1000000;

// #define TEST_SENDS
#define MEASURE

int main(int argc, char* argv[])
{
    UserMpi::MPI master(&argc, &argv);
    if (master.check()) return 1;

    master.setCommSize(MPI_COMM_WORLD);
    if (master.check()) return 1;

    master.setRank(MPI_COMM_WORLD);
    if (master.check()) return 1;
    
    if (master.getCommSize() != 2)
    {
        if (master.getRank() == 0)
            printf("Please run on two processes\n");
        return 0;
    }

#ifdef TEST_SENDS

    char* buffer = new char[max_size]{};

    if (master.getRank() == 1)
    {
        for (size_t i = 0; i < max_size; i++)
            buffer[i] = i;
    }

    double start = 0.f;
    double end   = 0.f;

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;

    if (master.getRank() != 0) 
    {
        start = MPI_Wtime();

        // Performs a blocking send
        master.send(buffer, max_size, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        end = MPI_Wtime();

        printf("MPI_Send: %.8lf\n", end - start);
    }
    else
    {
        sleep(2);

        master.recv(buffer, max_size, MPI_CHAR, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;

    if (master.getRank() != 0) 
    {
        start = MPI_Wtime();

        // Blocking synchronous send
        master.ssend(buffer, max_size, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        end = MPI_Wtime();

        printf("MPI_Ssend: %.8lf\n", end - start);
    }
    else
    {
        sleep(2);

        master.recv(buffer, max_size, MPI_CHAR, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;

    if (master.getRank() != 0) 
    {
        start = MPI_Wtime();

        // Blocking ready send
        master.rsend(buffer, max_size, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        end = MPI_Wtime();

        printf("MPI_Rsend: %.8lf\n", end - start);
    }
    else
    {
        sleep(2);

        master.recv(buffer, max_size, MPI_CHAR, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;

    if (master.getRank() != 0) 
    {
        int s = max_size + MPI_BSEND_OVERHEAD;
        char* b = new char[s];

        MPI_Buffer_attach(b, s);

        start = MPI_Wtime();

        //Basic send with user-provided buffering
        master.bsend(buffer, max_size, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        end = MPI_Wtime();

        MPI_Buffer_detach(&b, &s);

        printf("MPI_Bsend: %.8lf\n", end - start);
    }
    else
    {
        sleep(2);

        master.recv(buffer, max_size, MPI_CHAR, master.getRank() + 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;


#endif // TEST_SENDS
    

#ifdef MEASURE

    if (master.getRank() == 0)
    {
        printf("Size:      Time:\n");
    }

    char* buffer = new char[max_size]{};

    if (master.getRank() == 1)
    {
        for (size_t i = 0; i < max_size; i++)
            buffer[i] = i;
    }

    double means = 0.f;
    double start = 0.f;
    double end   = 0.f;

    master.barrier(MPI_COMM_WORLD);
    if (master.check()) return 1;

    for (size_t size = min_size; size < max_size; size += step)
    {
        means = 0.f;

        for (size_t i = 0; i < iters; i++)
        {   
            master.barrier(MPI_COMM_WORLD);
            if (master.check()) return 1;

            if (master.getRank() != 0) 
            {
                start = MPI_Wtime();

                master.send(buffer, size, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);

                end   = MPI_Wtime();
                means += end - start;
                if (master.check()) return 1;
            }
            else
            {
                // sleep(2);
                master.recv(buffer, size, MPI_CHAR, master.getRank() + 1, 0, MPI_COMM_WORLD);
                if (master.check()) return 1;
            }

            master.barrier(MPI_COMM_WORLD);
            if (master.check()) return 1;

        }

        if (master.getRank() != 0)
        {
            printf("%10lu %.10lf\n", size, means / iters);
        }

        master.barrier(MPI_COMM_WORLD);
        if (master.check()) return 1;
    }

    delete [] buffer;


#endif // MEASURE

    return 0;
}

