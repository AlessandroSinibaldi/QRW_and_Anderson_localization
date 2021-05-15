import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
from scipy import optimize
from pylab import *
import os 
import sys

filenames = ["Ljapunov_1.txt", "Ljapunov_1_5.txt", "Ljapunov_2.txt", "Ljapunov_2_5.txt", "Ljapunov_3.txt"]

N_energies = 500     # Se viene cambiata N_energies in Ljapunov_output.py, va cambiata anche qua
energies = np.zeros((len(filenames), N_energies))
localization_lenght = np.zeros((len(filenames), N_energies))

for k in range(0, len(filenames)):
    f = open(os.path.join(sys.path[0], filenames[k]), 'r')
    for j in range(0, N_energies):
        localization_lenght[k][j] = float(next(f))
        energies[k][j] = float(next(f))
    f.close()  

plt.figure()
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
title = "Esponente di Ljapunov"
plt.xlabel("Energia")
plt.ylabel(r'$1/\lambda_{1}$')
plt.grid(color='white', linestyle = 'dashed')
hfont = {'fontname':'monospace'}
plt.title(title)
plt.plot(energies[0], localization_lenght[0], label = "$\mathit{W}$ = 1")
plt.plot(energies[1], localization_lenght[1], label = "$\mathit{W}$ = 1.5")
plt.plot(energies[2], localization_lenght[2], label = "$\mathit{W}$ = 2")
plt.plot(energies[3], localization_lenght[3], label = "$\mathit{W}$ = 2.5")
plt.plot(energies[4], localization_lenght[4], label = "$\mathit{W}$ = 3")
plt.legend(loc = 'upper right')
plt.show()