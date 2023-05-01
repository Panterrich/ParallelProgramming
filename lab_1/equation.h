#ifndef EQUATION_H
#define EQUATION_H

#include <cmath>

namespace Equation 
{

static constexpr double a = 1.0;
static constexpr double X = 1.0;
static constexpr double T = 1.0;

/**
 * @brief The Courant — Friedrichs — Levy Criterion
 *        a * tau / h < C
 *        where C - Courant number
 * @note C = 1 for the cross method
 */
static constexpr double h = 0.05;
static constexpr double tau = 0.001;

static constexpr size_t K = T / tau;
static constexpr size_t M = X / h;

struct Func
{
    static double f(double x, double t)
    { 
        return std::exp(std::sin(x * t));
    }

    static double phi(double x)
    {
        return std::cos(M_PI * x);
    }

    static double psi(double t)
    {
        return std::exp(-t);
    }
};

}; // namespace Equation

#endif // EQUATION_H
