#ifndef EQUATION_H
#define EQUATION_H

#include <cmath>

namespace Equation 
{

/**
 * @brief The Courant — Friedrichs — Levy Criterion
 *        a * tau / h < C
 *        where C - Courant number
 * @note C = 1 for the cross method
 */

// /*NOTE - For fast/true mode */
// static constexpr double a = 1.0;
// static constexpr double X = 1.0;
// static constexpr double T = 1000.0;

// static constexpr double h = 0.001;
// static constexpr double tau = 0.01;

// /*NOTE - For slow/true mode */
// static constexpr double a = 1.0;
// static constexpr double X = 100.0;
// static constexpr double T = 10.0;

// static constexpr double h = 0.001;
// static constexpr double tau = 0.01;


// /*NOTE - For fast/false mode */
// static constexpr double a = 1.0;
// static constexpr double X = 500.0;
// static constexpr double T = 1.0;

// static constexpr double h = 0.005;
// static constexpr double tau = 0.001;

/*NOTE - For slow/false mode */
static constexpr double a = 1.0;
static constexpr double X = 5.0;
static constexpr double T = 100.0;

static constexpr double h = 0.005;
static constexpr double tau = 0.001;

static constexpr size_t K = T / tau;
static constexpr size_t M = X / h;

struct Func
{
    static double f(double x, double t)
    { 
        return std::exp(std::sin(x * t / X / T));
    }

    static double phi(double x)
    {
        return std::cos(M_PI * x / X);
    }

    static double psi(double t)
    {
        return std::exp(-t / T);
    }
};

}; // namespace Equation

#endif // EQUATION_H
