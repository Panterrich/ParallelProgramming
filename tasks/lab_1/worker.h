#ifndef WORKER_H
#define WORKER_H

#include <vector>
#include <memory.h>
#include <unistd.h>

#include "user_mpi.h"
#include "equation.h"

class Worker
{
public:
    explicit Worker(int rank, int commSize) :
        m_rank{rank},
        m_commSize{commSize},
        m_start{0},
        m_part{0},
        m_M{Equation::M},
        m_K{Equation::K},
        m_tau{Equation::tau},
        m_h{Equation::h},
        m_inversed{0},
        m_data{nullptr},
        m_result{nullptr}
    {
        SetPosition();

        m_data = new double[m_K * m_part]{};

        if (m_rank == 0)
        {   
            m_result = (m_commSize == 1) ? m_data : new double[m_K * m_M];
        }
    }

    ~Worker()
    {
        delete[] m_data;

        if (m_data != m_result)
        {
            delete[] m_result;
        }
    }

    int FillInitialConditions();

    int FillFirstLine(UserMpi::MPI* master);

    int FillOtherLines(UserMpi::MPI* master);

    int Dump(FILE* file);

    int Gather(UserMpi::MPI* master);

private:
    void SetPosition();

    int m_rank;
    int m_commSize;

    size_t m_start;
    size_t m_part;

    size_t m_M;
    size_t m_K;

    double m_tau;
    double m_h;

    int m_inversed;

    double* m_data;
    double* m_result;

}; // class Worker

#endif // WORKER_H
