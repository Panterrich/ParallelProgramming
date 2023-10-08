#ifndef EQUATION_H
#define EQUATION_H

#include <cmath>

namespace Equation 
{

struct Func
{
    static double f(double x)
    { 
        return std::cos(1 / (x * x));
        //return std::sin(1 / (x + 20));
    }

    static constexpr double a = 0.01f;
    static constexpr double b = 1.f;

    // static constexpr double a = -19.99f;
    // static constexpr double b =  10.f;
};

}; // namespace Equation

#endif // EQUATION_H
