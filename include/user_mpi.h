#ifndef USER_MPI_H
#define USER_MPI_H

#include <stdio.h>
#include <err.h>
#include <mpi.h>

namespace UserMpi
{

class MPI
{
public:
    MPI() = delete;
    MPI(int* argc, char** argv[]):
        m_commSize{},
        m_rank{},
        m_buffer{},
        m_len{},
        m_error{}
    {
        setbuf(stdout, nullptr);

        m_error = MPI_Init(argc, argv);
    }

    MPI(const MPI& mpi) = delete;

    ~MPI()
    {
        MPI_Finalize();
    }

    int check() 
    {
        int error = getError(); 

        if (error)
        {
            MPI_Error_string(error, m_buffer, &m_len);
            warn("MPI: %s\n", m_buffer);
        }

        return error;
    }

    inline int getError() const
    {
        return m_error;
    }

    inline const char* getErrorString()
    {
        MPI_Error_string(getError(), m_buffer, &m_len);

        return m_buffer;
    }

    inline int getCommSize() const
    {
        return m_commSize;
    }

    inline int getRank() const
    {
        return m_rank;
    }

    inline void setCommSize(MPI_Comm comm)
    {
        m_error = MPI_Comm_size(comm, &m_commSize);
    }

    inline void setRank(MPI_Comm comm)
    {
        m_error = MPI_Comm_rank(comm, &m_rank);
    }

    inline void send(const void* buffer, int count, MPI_Datatype type, int dst, int tag, MPI_Comm comm)
    {
        m_error = MPI_Send(buffer, count, type, dst, tag, comm);
    }

    inline void recv(void* buffer, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm)
    {
        m_error = MPI_Recv(buffer, count, type, src, tag, comm, &m_status);
    }

    inline void probe(int src, int tag, MPI_Comm comm)
    {
        m_error = MPI_Probe(src, tag, comm, &m_status);
    }

    inline void reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    {
        m_error = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    }

    inline int getCount(MPI_Datatype datatype)
    {
        int count = 0;

        m_error = MPI_Get_count(&m_status, datatype, &count);

        return count;
    }

    inline const MPI_Status& getStatus() const
    {
        return m_status;
    }

private:
    int  m_commSize;
    int  m_rank;

    MPI_Status m_status;

    char m_buffer[MPI_MAX_ERROR_STRING];
    int  m_len;

    int  m_error;
    
}; // class MPI

} // UserMpi

#endif // USER_MPI_H
