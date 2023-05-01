#include "worker.h"

void Worker::SetPosition()
{
    m_M     = (m_M / m_commSize + !!(m_M % m_commSize)) * m_commSize;
    m_part  =  m_M / m_commSize;
    m_start =  m_part * m_rank;
}

int Worker::FillInitialConditions()
{
    if (m_rank == 0) 
    {
        for (size_t i = 0; i < m_K; i++) 
        {
            m_data[i * m_part] = Equation::Func::psi(Equation::tau * i);
        }
    }

    for (size_t i = 0; i < m_part; i++) 
    {
        m_data[i] = Equation::Func::phi(Equation::h * (m_start + i));
    }

    return 0;
}

int Worker::FillFirstLine(UserMpi::MPI* master)
{
    double up_value = 0;
    double down_value = 0;

    if (m_rank != 0)
    {
        master->recv(&up_value,   1, MPI::DOUBLE, m_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master->check()) return 1;
        
        master->recv(&down_value, 1, MPI::DOUBLE, m_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master->check()) return 1;
    }

    for (size_t m = (m_rank == 0); m < m_part; m++) 
    {
        if (m != 0)
        {
            up_value   = m_data[m_part + m - 1];
            down_value = m_data[         m - 1];
        }

        double first_part  =               ( up_value - down_value - m_data[m]) / (2 * Equation::tau);
        double second_part = Equation::a * (-up_value - down_value + m_data[m]) / (2 * Equation::h);

        double f_part  = Equation::Func::f((m_start + m + 0.5) * Equation::h, (1 + 0.5) * Equation::tau);

        m_data[m_part + m] = (f_part - first_part - second_part) * 2 * Equation::tau * Equation::h / (Equation::tau + Equation::h);
    }

    if (m_rank != m_commSize - 1)
    {
        master->send(m_data + m_part + m_part - 1, 1, MPI::DOUBLE, m_rank + 1, 0, MPI_COMM_WORLD);
        if (master->check()) return 1;

        master->send(m_data + m_part - 1, 1, MPI::DOUBLE, m_rank + 1, 0, MPI_COMM_WORLD);
        if (master->check()) return 1;
    }

    return 0;
}

int Worker::FillOtherLines(UserMpi::MPI* master)
{
    double recv_value = 0;

    for (size_t k = 1; k < m_K - 1; k++) 
    {
        if (m_rank != m_commSize - 1) 
        {
            master->send(m_data + m_part * k + m_part - 1, 1, MPI::DOUBLE, m_rank + 1, 0, MPI_COMM_WORLD);
            if (master->check()) return 1;
        }

        if (m_rank != 0) 
        {
            master->recv(&recv_value, 1, MPI::DOUBLE, m_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD);
            if (master->check()) return 1;
        }

        for (size_t m = (m_rank == 0); m < m_part - 1; m++) 
        {
            if (m != 0) 
            {
                recv_value = m_data[m_part * k + m - 1];
            }

            double first_part  =               (- m_data[m_part * (k - 1) + m]                 ) / (2 * Equation::tau);
            double second_part = Equation::a * (  m_data[m_part *  k      + m + 1] - recv_value) / (2 * Equation::h);

            double func_part   = Equation::Func::f((m_start + m) * Equation::h, k * Equation::tau);

            m_data[m_part * (k + 1) + m] = (func_part - first_part - second_part) * 2 * Equation::tau;
        }

        size_t m = m_part - 1;

        double first_part  =               (  m_data[m_part * (k + 1) + m - 1] - m_data[m_part * k + m - 1] - m_data[m_part * k + m]) / (2 * Equation::tau);
        double second_part = Equation::a * (- m_data[m_part * (k + 1) + m - 1] - m_data[m_part * k + m - 1] + m_data[m_part * k + m]) / (2 * Equation::h);

        double func_part = Equation::Func::f((m_start + m + 0.5) * Equation::h, (k + 0.5) * Equation::tau);

        m_data[m_part * (k + 1) + m] = (func_part - first_part - second_part) * 2 * Equation::tau * Equation::h / (Equation::tau + Equation::h);
    }

    return 0;
}

int Worker::Dump(FILE* file)
{
    fprintf(file, "X:   %lg\n", Equation::X);
    fprintf(file, "h:   %lg\n", Equation::h);
    fprintf(file, "T:   %lg\n", Equation::T);
    fprintf(file, "tau: %lg\n", Equation::tau);
    fprintf(file, "M:   %lu\n", Equation::M);
    fprintf(file, "K:   %lu\n", Equation::K);

    for (size_t i = 0; i < Equation::K * Equation::M; i++)
    {
        fprintf(file, "%lg\n", m_result[i]);
    }

    return 0;
}

int Worker::Gather(UserMpi::MPI* master)
{    
    for (size_t k = 0; k < m_K; k++) 
    {
        master->gather(m_data + k * m_part, m_part, MPI::DOUBLE, m_result + k * m_M, m_part, MPI::DOUBLE, 0, MPI_COMM_WORLD);
        if (master->check()) return 1;
    }

    return 0;
}