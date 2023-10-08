#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import shutil
import numpy as np
from matplotlib import pyplot as plt

executable = './build/lab_2'
max_proc = 12
n_tests  = 6
eps = " 1e-8 "

default_output_dir = os.path.join(os.path.split(os.path.dirname(os.path.relpath(os.path.realpath(__file__))))[0], 'output/')

def main():
    parser = argparse.ArgumentParser(description='Run plotting')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help='Echo system command or not')

    subparsers = parser.add_subparsers(help='targets', dest='target')

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

    if target == 'time':
        TimeGraph(quiet)

def TimeGraph(quiet):
    log = os.path.join(default_output_dir, 'output.txt')

    y = []

    for i in range(1, max_proc + 1):

        data = 0

        for j in range(n_tests):
            
            command = executable + " " + str(i) + eps + " > " + log

            if not quiet:
                print(command)

            os.system(command)

            x = 0
            with open(log) as f:
                for line in f:
                    line = line.split()
                    time = line[1]
                    if x == i + 1:
                        data += float(time)
                        break
                    x += 1
        
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
    plt.savefig(os.path.join(default_output_dir, "output.png"))
    plt.show()


if __name__ == "__main__":
    main()
