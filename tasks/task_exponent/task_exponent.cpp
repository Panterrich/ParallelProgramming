#include <cmath>
#include <gmp.h>

#include "user_mpi.h"

unsigned long AccuracyToN(unsigned long x);

int main(int argc, char* argv[])
{
    if (argc != 2) 
    {
        printf("Enter with what accuracy to calculate the exponent\n"
               "The number of digits after the point.\n"
               "For example: ./a.out 10000\n");
        return 0;
    }

    char* end = nullptr;
    unsigned long x = strtoul(argv[1], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    unsigned long N = AccuracyToN(x);

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
        beginPoint = N -  master.getRank()      * (sectionSize + 1);
          endPoint = N - (master.getRank() + 1) * (sectionSize + 1);
    }
    else
    {
        beginPoint = N - ( master.getRank()      * sectionSize + remains);
          endPoint = N - ((master.getRank() + 1) * sectionSize + remains);
    }

    mpf_set_default_prec(64 + std::ceil(8 * x));

    mpz_t fuc = {};
    mpz_t sum = {};

    mpz_init_set_ui(fuc, 1);
    mpz_init_set_ui(sum, 0);
    
    for (unsigned long i = beginPoint; i > endPoint; i--)
    {
        mpz_mul_ui(fuc, fuc, i);
        mpz_add(sum, sum, fuc);
    }

    if (master.getRank() < master.getCommSize() - 1)
    {
        master.probe(master.getRank() + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        int count = master.getCount(MPI_CHAR);
        if (master.check()) return 1;

        char* sum_str = new char[count]();

        master.recv(sum_str, count, MPI_CHAR, master.getRank() + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        master.probe(master.getRank() + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        count = master.getCount(MPI_CHAR);
        if (master.check()) return 1;

        char* fuc_str = new char[count]();

        master.recv(fuc_str, count, MPI_CHAR, master.getRank() + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
        if (master.check()) return 1;

        mpz_t sum_recv = {};
        mpz_t fuc_recv = {};

        mpz_init_set_str(sum_recv, sum_str, 10);
        mpz_init_set_str(fuc_recv, fuc_str, 10);

        delete [] sum_str;
        delete [] fuc_str;

        mpz_mul(sum_recv, sum_recv, fuc);
        mpz_mul(fuc, fuc, fuc_recv);
        mpz_add(sum, sum, sum_recv);

        mpz_clears(sum_recv, fuc_recv, nullptr);
    }

    if (master.getRank() > 0)
    {
        char* sum_str = mpz_get_str(nullptr, 10, sum);
        char* fuc_str = mpz_get_str(nullptr, 10, fuc);

        unsigned long sum_len = strlen(sum_str) + 1;
        unsigned long fuc_len = strlen(fuc_str) + 1;

        master.send(sum_str, sum_len, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        master.send(fuc_str, fuc_len, MPI_CHAR, master.getRank() - 1, 0, MPI_COMM_WORLD);
        if (master.check()) return 1;

        free(sum_str);
        free(fuc_str);

        mpz_clears(fuc, sum, nullptr);
    }

    else
    {
        mpz_add_ui(sum, sum, 1);

        mpf_t sumf = {};
        mpf_t fucf = {};
        
        mpf_init(sumf);
        mpf_init(fucf);

        mpf_set_z(sumf, sum);
        mpf_set_z(fucf, fuc);

        mpf_div(sumf, sumf, fucf);

        gmp_printf("%.*Ff\n", x, sumf);

        mpz_clears(fuc,  sum,  nullptr);
        mpf_clears(fucf, sumf, nullptr);
    }

    return 0;
}


/**
 * @brief AccuracyToN - calculates the estimate of the number N (maximum factorial size) 
 * 
 * Need: N! > 10^x
 * Let 's use the Stirling formula: ln(N!) > N * ln(N) - N > ln(10 ^ x) = x * ln(10)
 * To solve this equation, we use Newton's method
 *  
 * @param x - accuracy
 * @return N 
 */
unsigned long AccuracyToN(unsigned long x)
{
    const double a   = std::log(10) * x;
    const double eps = 1.0;

    double N_0 = 2.0;
    double N   = 0;

    double log_N = 0;

    while (true)
    {   
        log_N = std::log(N_0);
        N = N_0 - (N_0 * log_N - N_0 - a) / log_N;

        if (std::abs(N - N_0) < eps)
            break;

        N_0 = N;
    }

    return std::ceil(N);
}
