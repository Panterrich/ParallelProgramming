#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import shutil
import numpy as np
from matplotlib import pyplot as plt

executable = './build/lab_1'
max_proc = 6
n_tests  = 6

default_output_dir = os.path.join(os.path.split(os.path.dirname(os.path.relpath(os.path.realpath(__file__))))[0], 'output/')

def main():
    parser = argparse.ArgumentParser(description='Run plotting')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help='Echo system command or not')

    subparsers = parser.add_subparsers(help='targets', dest='target')

    subparsers.add_parser('solution', help='Solution graph')
    subparsers.add_parser('time', help='Time graph')

    args = parser.parse_args()

    target = args.target
    quiet  = args.quiet

    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    else:
        if not os.path.isdir(default_output_dir):
            print("Specify the output directory!")
            return 0

    if target == 'solution':
        SolutionGraph(quiet)

    elif target == 'time':
        TimeGraph(quiet)


def SolutionGraph(quiet):
    log = os.path.join(default_output_dir, 'output.txt')

    command = "mpirun -np " + str(max_proc) + " " + executable + " " + log

    if not quiet:
        print(command)

    os.system(command)

    params = {
    "X": 0,
    "h": 0,
    "T": 0,
    "tau": 0,
    "M": 0,
    "K": 0
    }

    data = []

    with open(log) as f:
        for line in f:
            line = line.split()
            name = line[0]

            if name[:-1] in params.keys():
                params[name[:-1]] = float(line[1])
            else:
                data.append(float(name))
      
    U = np.zeros((int(params['K']), int(params['M'])))

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i][j] = data[i * U.shape[1] + j]

    X = np.array([k * params['h'] for k in range(U.shape[1])])
    T = np.array([k * params['tau'] for k in range(U.shape[0])])

    X, T = np.meshgrid(X, T)

    fg = plt.figure(figsize=(22, 15))
    ax = fg.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, U, cmap='coolwarm')
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$t$", fontsize=20)
    ax.set_zlabel("$u(x, t)$", fontsize=20)
    plt.savefig(os.path.join(default_output_dir, "output.png"))
    plt.show()

def TimeGraph(quiet):
    log = os.path.join(default_output_dir, 'output.txt')

    y = []

    for i in range(1, max_proc + 1):

        data = 0

        for j in range(n_tests):
            
            command = "mpirun -np " + str(i) + " " + executable + " > " + log

            if not quiet:
                print(command)

            os.system(command)

            with open(log) as f:
                for line in f:
                    line = line.split()
                    time = line[1]

                    data += float(time)
        
        data /= n_tests
        y.append(data)

    plt.figure(figsize = (16, 16), facecolor = "white") # Создаем фигуру
    plt.style.use('default')

    plt.title(r'S(p)')
    plt.ylabel(r'$S$')
    plt.xlabel(r"n proc")

    x = np.arange(1, max_proc + 1, 1)


    for i in range(1, max_proc):
        y[i] = y[0] / y[i]

    y[0] = 1

    plt.plot(x, y)

    plt.grid(visible = True, which = 'major', axis = 'both', alpha = 1, linewidth = 0.9)   # Активируем сетку
    plt.grid(visible = True, which = 'minor', axis = 'both', alpha = 0.5, linestyle = ':')

    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
            

if __name__ == "__main__":
    main()
