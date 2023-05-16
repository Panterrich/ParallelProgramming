#ifndef LAB_2_H
#define LAB_2_H

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stack>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdlib.h>
#include <errno.h>

#include "equation.h"

void* routine_integrate(void* arg);

void InitSharedMemory(double eps);

void DestroySharedMemory();

struct Task
{
    double a;
    double b;

    double fa;
    double fb;

    double s;

    Task() :
        a{Equation::Func::a},
        b{Equation::Func::b},
        fa{Equation::Func::f(a)},
        fb{Equation::Func::f(b)},
        s{(fb + fa) / 2 * (b - a)}
    {}

    Task(double a_, double b_, double fa_, double fb_, double s_):
        a{a_},
        b{b_},
        fa{fa_},
        fb{fb_},
        s{s_}
    {}
};

#endif // LAB_2_H