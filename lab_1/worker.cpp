#include "worker.h"

void Worker::SetPosition()
{
    if (Equation::a * Equation::tau / Equation::h < 1)
    {
        if (m_rank == 0)
        {
            if (m_M < m_K)
            {
                printf("Mode: slow\nInversed: false\n");
            }
            else 
            {
                printf("Mode: fast\nInversed: false\n");
            }
        }

        m_inversed = 0;
    }
    else
    {
        if (m_rank == 0)
        {
            if (m_M < m_K)
            {
                printf("Mode: fast\nInversed: true\n");
            }
            else 
            {
                printf("Mode: slow\nInversed: true\n");
            }
        }

        std::swap(m_M, m_K);
        std::swap(m_h, m_tau);
    
        m_inversed = 1;

    }

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
            m_data[i * m_part] = m_inversed ? Equation::Func::phi(m_tau * i) : 
                                              Equation::Func::psi(m_tau * i);
        }
    }

    for (size_t i = 0; i < m_part; i++) 
    {
        m_data[i] = m_inversed ? Equation::Func::psi(m_h * (m_start + i)) : 
                                 Equation::Func::phi(m_h * (m_start + i));
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

        double first_part  = ( up_value - down_value - m_data[m]) / (2 * m_tau);
        double second_part = (-up_value - down_value + m_data[m]) / (2 * m_h);

        double f_part  = m_inversed ? Equation::Func::f((1 + 0.5) * m_tau, (m_start + m + 0.5) * m_h) :
                                      Equation::Func::f((m_start + m + 0.5) * m_h, (1 + 0.5) * m_tau);

        m_data[m_part + m] = m_inversed ? 
            (f_part - Equation::a * first_part -               second_part) * 2 / (Equation::a / m_tau +           1 / m_h) :
            (f_part -               first_part - Equation::a * second_part) * 2 / (          1 / m_tau + Equation::a / m_h);
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

            double first_part  = (- m_data[m_part * (k - 1) + m]                 ) / (2 * m_tau);
            double second_part = (  m_data[m_part *  k      + m + 1] - recv_value) / (2 * m_h);

            double f_part = m_inversed ? Equation::Func::f(k * m_tau, (m_start + m) * m_h) :
                                         Equation::Func::f((m_start + m) * m_h, k * m_tau);

            m_data[m_part * (k + 1) + m] = m_inversed ? (f_part - Equation::a * first_part -               second_part) * 2 * m_tau / Equation::a :
                                                        (f_part -               first_part - Equation::a * second_part) * 2 * m_tau;
        }

        size_t m = m_part - 1;

        double first_part  = (  m_data[m_part * (k + 1) + m - 1] - m_data[m_part * k + m - 1] - m_data[m_part * k + m]) / (2 * m_tau);
        double second_part = (- m_data[m_part * (k + 1) + m - 1] - m_data[m_part * k + m - 1] + m_data[m_part * k + m]) / (2 * m_h);

        double f_part = m_inversed ? Equation::Func::f((k + 0.5) * m_tau, (m_start + m + 0.5) * m_h) :
                                     Equation::Func::f((m_start + m + 0.5) * m_h, (k + 0.5) * m_tau);

        m_data[m_part * (k + 1) + m] = m_inversed ? 
            (f_part - Equation::a * first_part -               second_part) * 2 / (Equation::a / m_tau +           1 / m_h) :
            (f_part -               first_part - Equation::a * second_part) * 2 / (          1 / m_tau + Equation::a / m_h);
    }

    return 0;
}

int Worker::Dump(FILE* file)
{
    if (m_inversed)
    {
        std::swap(m_M, m_K);
        std::swap(m_h, m_tau);
    }

    fprintf(file, "X:   %lg\n", Equation::X);
    fprintf(file, "h:   %lg\n", m_h);
    fprintf(file, "T:   %lg\n", Equation::T);
    fprintf(file, "tau: %lg\n", m_tau);
    fprintf(file, "M:   %lu\n", Equation::M);
    fprintf(file, "K:   %lu\n", Equation::K);

    for (size_t i = 0; i < m_K; i++)
    {   
        if (i >= Equation::K) break;

        for (size_t j = 0; j < m_M; j++)
        {
            if (j >= Equation::M) break;

            if (m_inversed)
            {
                fprintf(file, "%lg\n", m_result[j * m_K + i]);   
            }
            else
            {
                fprintf(file, "%lg\n", m_result[i * m_M + j]);
            }
        }
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